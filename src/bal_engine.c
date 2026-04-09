#include "bal_engine.h"
#include "bal_decoder.h"
#include "bal_logging.h"
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#define MAX_INSTRUCTIONS 65536

// Not sure what exact value to put here.
//
#define MAX_GUEST_REGISTERS 128

/// The size of an ARM Instruction in bytes.
#define ARM_INSTRUCTION_SIZE_BYTES 4

/// Helper macro to align `x` UP to the nearest memory alignment.
#define BAL_ALIGN_UP(x, memory_alignment) \
    (((x) + ((memory_alignment) - 1)) & ~((memory_alignment) - 1))

typedef struct
{
    bal_instruction_t      *ir_instruction_cursor;
    bal_bit_width_t        *bit_width_cursor;
    bal_source_variable_t  *source_variables;
    bal_constant_t         *constants;
    size_t                  constants_size;
    bal_constant_count_t    constant_count;
    bal_instruction_count_t instruction_count;
    bal_error_t             status;
    bal_logger_t           *logger;
} bal_translation_context_t;

static uint32_t extract_operand_value(uint32_t, const bal_decoder_operand_t *);
static uint32_t intern_constant(bal_translation_context_t *, bal_constant_t);
static void     translate_add(bal_translation_context_t                *context,
                              const bal_decoder_instruction_metadata_t *metadata,
                              const uint32_t                           *arm_registers);
static void     translate_const(bal_translation_context_t *,
                                const bal_decoder_instruction_metadata_t *,
                                const uint32_t *);
static void     translate_return(bal_translation_context_t *, const uint32_t *);
BAL_COLD bal_error_t
bal_engine_init(const bal_allocator_t *allocator, bal_engine_t *engine, bal_logger_t logger)
{
    if (NULL == allocator || NULL == engine)
    {
        return BAL_ERROR_INVALID_ARGUMENT;
    }

    const size_t source_variables_size = MAX_GUEST_REGISTERS * sizeof(bal_source_variable_t);
    const size_t ssa_bit_widths_size   = MAX_INSTRUCTIONS * sizeof(bal_bit_width_t);
    const size_t instructions_size     = MAX_INSTRUCTIONS * sizeof(bal_instruction_t);
    const size_t constants_size        = MAX_INSTRUCTIONS * sizeof(bal_instruction_t);

    // Calculate amount of memory needed for all arrays in engine.
    //
    const size_t memory_alignment    = 64U;
    const size_t offset_instructions = BAL_ALIGN_UP(source_variables_size, memory_alignment);

    const size_t offset_ssa_bit_widths
        = BAL_ALIGN_UP((offset_instructions + instructions_size), memory_alignment);

    const size_t offset_constants
        = BAL_ALIGN_UP((offset_ssa_bit_widths + ssa_bit_widths_size), memory_alignment);

    const size_t total_size_with_padding
        = BAL_ALIGN_UP((offset_constants + constants_size), memory_alignment);

    uint8_t *data
        = allocator->allocate(allocator->handle, memory_alignment, total_size_with_padding);

    BAL_LOG_DEBUG(&logger, "Calculating arena layout (Alignment: %zu bytes):", memory_alignment);
    BAL_LOG_DEBUG(
        &logger, "  [0x%08zx] source_variables (%zu bytes)", (size_t)0, source_variables_size);
    BAL_LOG_DEBUG(&logger,
                  "  [0x%08zx] instructions     (%zu bytes)",
                  offset_instructions,
                  instructions_size);
    BAL_LOG_DEBUG(&logger,
                  "  [0x%08zx] ssa_bit_widths   (%zu bytes)",
                  offset_ssa_bit_widths,
                  ssa_bit_widths_size);
    BAL_LOG_DEBUG(
        &logger, "  [0x%08zx] constants        (%zu bytes)", offset_constants, constants_size);

    if (NULL == data)
    {
        BAL_LOG_ERROR(&logger, "Allocation of %zu bytes failed.", total_size_with_padding);
        engine->status = BAL_ERROR_ALLOCATION_FAILED;
        return engine->status;
    }

    engine->source_variables      = (bal_source_variable_t *)data;
    engine->instructions          = (bal_instruction_t *)(data + offset_instructions);
    engine->constants             = (bal_constant_t *)(data + offset_constants);
    engine->ssa_bit_widths        = data + offset_ssa_bit_widths;
    engine->source_variables_size = source_variables_size / sizeof(bal_source_variable_t);
    engine->instructions_size     = instructions_size / sizeof(bal_instruction_t);
    engine->constants_size        = constants_size / sizeof(bal_constant_t);
    engine->constant_count        = 0;
    engine->instruction_count     = 0;
    engine->status                = BAL_SUCCESS;
    engine->arena_base            = (void *)data;
    engine->arena_size            = total_size_with_padding;
    engine->logger                = logger;

    BAL_LOG_INFO(&logger,
                 "Initialized engine successfully. Arena: %p (%zu KB)",
                 engine->arena_base,
                 total_size_with_padding / 1024);

    (void)memset(engine->source_variables, POISON_UNINITIALIZED_MEMORY, source_variables_size);
    (void)memset(engine->instructions, POISON_UNINITIALIZED_MEMORY, instructions_size);
    (void)memset(engine->ssa_bit_widths, POISON_UNINITIALIZED_MEMORY, ssa_bit_widths_size);
    (void)memset(engine->constants, POISON_UNINITIALIZED_MEMORY, constants_size);

    return engine->status;
}

bal_error_t
bal_engine_translate(bal_engine_t *BAL_RESTRICT                 engine,
                     const bal_memory_interface_t *BAL_RESTRICT interface,
                     bal_guest_address_t                       *guest_address_start,
                     const size_t                               max_instructions)
{
    if (BAL_UNLIKELY(NULL == engine))
    {
        return BAL_ERROR_INVALID_ARGUMENT;
    }

    if (BAL_UNLIKELY(engine->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&engine->logger, "Engine != BAL_SUCCESS, aborting translation");
        return BAL_ERROR_INVALID_ARGUMENT;
    }

    if (BAL_UNLIKELY(NULL == interface))
    {
        BAL_LOG_ERROR(&engine->logger, "Interface is NULL, aborting translation");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(0 == max_instructions))
    {
        BAL_LOG_INFO(&engine->logger, "Max Instructions is 0, aborting translation");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == &guest_address_start))
    {
        BAL_LOG_INFO(&engine->logger, "Guest Address is NULL, assuming entry point is 0x0");
    }

    const size_t max_instructions_size_bytes = max_instructions * ARM_INSTRUCTION_SIZE_BYTES;
    BAL_LOG_INFO(&engine->logger,
                 "Starting JIT unit. GVA: %p, Size: %zu bytes ",
                 (void *)*guest_address_start,
                 max_instructions_size_bytes);

    bal_translation_context_t context
        = { .ir_instruction_cursor = engine->instructions + engine->instruction_count,
            .bit_width_cursor      = engine->ssa_bit_widths + engine->instruction_count,
            .source_variables      = engine->source_variables,
            .constants             = engine->constants,
            .constants_size        = engine->constants_size,
            .constant_count        = engine->constant_count,
            .instruction_count     = engine->instruction_count,
            .status                = engine->status,
            .logger                = &engine->logger };

    bool     is_block_terminated                         = false;
    uint32_t arm_instruction_operands[BAL_OPERANDS_SIZE] = { 0 };

    while (false == is_block_terminated)
    {
        size_t                            max_readable_instructions_bytes = 0;
        bal_guest_address_t *BAL_RESTRICT guest_address_current           = guest_address_start;

        if (BAL_UNLIKELY((uintptr_t)guest_address_current % 4 != 0))
        {
            BAL_LOG_ERROR(context.logger,
                          "Guest Virtual Address 0x%016llX is not 4-byte aligned",
                          (unsigned long long)guest_address_current);
            context.status = BAL_ERROR_MEMORY_ALIGNMENT;
            break;
        }
        const uint32_t *host_address_base = (const uint32_t *)interface->translate(
            (void *)interface, *guest_address_current, &max_readable_instructions_bytes);

        if (BAL_UNLIKELY(NULL == host_address_base))
        {
            BAL_LOG_ERROR(context.logger,
                          "Memory translation fault at GVA 0x%016llX, aborting translation block",
                          (unsigned long long)*guest_address_start);
            context.status = BAL_ERROR_MEMORY_FAULT;
            break;
        }

        const uint32_t *BAL_RESTRICT host_address_current = host_address_base;
        const size_t max_readable_instructions = max_readable_instructions_bytes / sizeof(uint32_t);

        // Has the remaining instructions crossed a memory page boundary? This should not happen on
        // ARM64 since instructions are strictly 4 byte aligned. So throw an error if this
        // happens.
        //
        if (BAL_UNLIKELY(0 == max_readable_instructions))
        {
            BAL_LOG_ERROR(context.logger,
                          "Insufficient memory at GVA 0x%016llX. Need 4 bytes for an instruction, "
                          "but only %zu bytes are readable",
                          (unsigned long long)guest_address_current,
                          max_readable_instructions_bytes);
            context.status = BAL_ERROR_PC_ALIGNMENT;
            break;
        }

        for (size_t i = 0; i < max_readable_instructions; ++i)
        {
            if (BAL_UNLIKELY(context.instruction_count >= (MAX_INSTRUCTIONS - 128)))
            {
                BAL_LOG_WARN(context.logger,
                             "Critical buffer pressure. Inst:  %u/%d",
                             context.instruction_count,
                             MAX_INSTRUCTIONS);
            }

            const bal_decoder_instruction_metadata_t *metadata
                = bal_decode_arm64(*host_address_current);

            const size_t relative_offset
                = (uintptr_t)host_address_current - (uintptr_t)host_address_base;

            if (BAL_UNLIKELY(NULL == metadata))
            {
                BAL_LOG_ERROR(context.logger,
                              "Decode failed for GVA 0x%08x at offset +0x%zx",
                              *guest_address_start,
                              relative_offset);
                context.status      = BAL_ERROR_UNKNOWN_INSTRUCTION;
                is_block_terminated = true;
                break;
            }

            BAL_LOG_DEBUG(context.logger,
                          "  [+0x%04zx] 0x%08x: %-8s (SSA ID: %u)",
                          relative_offset,
                          *guest_address_start,
                          metadata->name,
                          context.instruction_count);

            const bal_decoder_operand_t *BAL_RESTRICT operands_cursor = metadata->operands;

            for (size_t ii = 0; ii < BAL_OPERANDS_SIZE; ++ii)
            {
                arm_instruction_operands[ii]
                    = extract_operand_value(*host_address_current, operands_cursor);
                ++operands_cursor;
            }

            switch (metadata->ir_opcode)
            {
                case OPCODE_ADD:
                    translate_add(&context, metadata, arm_instruction_operands);
                    break;
                case OPCODE_CONST:
                    translate_const(&context, metadata, arm_instruction_operands);
                    break;
                case OPCODE_RETURN:
                    translate_return(&context, arm_instruction_operands);
                    is_block_terminated = true;
                    break;
                default:
                    BAL_LOG_DEBUG(context.logger,
                                  "  SKIPPED: Opcode %s not implemented in IR layer.",
                                  metadata->name);
                    break;
            }

            if (BAL_UNLIKELY(context.status != BAL_SUCCESS))
            {
                BAL_LOG_ERROR(context.logger, "  Status failure: %d", context.status);
                is_block_terminated = true;
                break;
            }

            if (true == is_block_terminated)
            {
                break;
            }

            ++context.ir_instruction_cursor;
            ++context.bit_width_cursor;
            *guest_address_current += 4;
            ++host_address_current;
        }
    }

    engine->instruction_count = context.instruction_count;
    engine->constant_count    = context.constant_count;
    engine->status            = context.status;

    BAL_LOG_INFO(&engine->logger,
                 "Finished. Produced %u instructions, %u constants.",
                 engine->instruction_count,
                 engine->constant_count);

    return engine->status;
}

bal_error_t
bal_engine_reset(bal_engine_t *engine)
{
    if (BAL_UNLIKELY(NULL == engine))
    {
        return BAL_ERROR_INVALID_ARGUMENT;
    }

    engine->instruction_count = 0;
    engine->status            = BAL_SUCCESS;

    (void)memset(
        engine->source_variables, POISON_UNINITIALIZED_MEMORY, engine->source_variables_size);

    (void)memset(engine->constants, POISON_UNINITIALIZED_MEMORY, engine->constants_size);

    return engine->status;
}

void
bal_engine_destroy(const bal_allocator_t *allocator, bal_engine_t *engine)
{
    // No argument error handling. Segfault if user passes NULL.

    allocator->free(allocator->handle, engine->arena_base, engine->arena_size);
    engine->arena_base       = NULL;
    engine->source_variables = NULL;
    engine->instructions     = NULL;
    engine->ssa_bit_widths   = NULL;
}

BAL_HOT static uint32_t
extract_operand_value(const uint32_t instruction, const bal_decoder_operand_t *operand)
{
    if (BAL_OPERAND_TYPE_NONE == operand->type)
    {
        return 0;
    }

    const uint32_t mask = (1U << operand->bit_width) - 1;
    const uint32_t bits = instruction >> operand->bit_position & mask;
    return bits;
}

// If `constant` exists in the constants array return its index, if not, add it and then return
// its index
BAL_HOT static uint32_t
intern_constant(bal_translation_context_t *BAL_RESTRICT context, const bal_constant_t constant)
{
    if (BAL_UNLIKELY(context->status != BAL_SUCCESS))
    {
        return 0;
    }

    // This can be upgraded to a hash map later on if it causes performance loss.
    //
    for (uint32_t i = 0; i < context->constant_count; ++i)
    {
        if (context->constants[i] == constant)
        {
            BAL_LOG_TRACE(context->logger, "  0X%016llX -> c%u", (unsigned long long)constant, i);
            return i | BAL_IS_CONSTANT_BIT_POSITION;
        }
    }

    const uint32_t index = context->constant_count;

    if (BAL_UNLIKELY(index >= context->constants_size))
    {
        BAL_LOG_ERROR(context->logger,
                      "Constant Pool Overflow at index %u (Max %u)",
                      index,
                      context->constants_size);
        context->status = BAL_ERROR_INSTRUCTION_OVERFLOW;
        return 0;
    }

    context->constants[index] = constant;
    context->constant_count++;
    BAL_LOG_TRACE(context->logger, "  0X%016llX -> c%u", (unsigned long long)constant, index);
    return index | BAL_IS_CONSTANT_BIT_POSITION;
}

BAL_HOT static uint32_t
get_or_create_ssa_index(bal_translation_context_t *context, const uint64_t register_index)
{
    uint32_t ssa_index = context->source_variables[register_index].current_ssa_index;

    const uint32_t invalid_ssa_index = 0xFFFFFFFF;

    // Checks if the ssa index is not initialized.
    //
    if (ssa_index != invalid_ssa_index)
    {
        return ssa_index;
    }

    const bal_instruction_t instruction = (bal_instruction_t)OPCODE_GET_REGISTER
                                              << BAL_OPCODE_SHIFT_POSITION
                                          | register_index << BAL_SOURCE1_SHIFT_POSITION;

    *context->ir_instruction_cursor                             = instruction;
    ssa_index                                                   = context->instruction_count;
    context->source_variables[register_index].current_ssa_index = ssa_index;

    BAL_LOG_DEBUG(context->logger,
                  "  EMIT: v%u = GET_REGISTER X%u",
                  context->instruction_count,
                  register_index);
    BAL_LOG_TRACE(
        context->logger, "  SSA UPDATE: X%lu -> v%lu", register_index, context->instruction_count);

    context->instruction_count++;
    context->ir_instruction_cursor++;
    context->bit_width_cursor++;
    return ssa_index;
}

BAL_HOT static void
translate_add(bal_translation_context_t                *context,
              const bal_decoder_instruction_metadata_t *metadata,
              const uint32_t                           *arm_registers)
{
    const uint64_t rd = arm_registers[0];
    const uint64_t rn = arm_registers[1];
    const uint64_t sh = arm_registers[3];

    // Add Immediate instruction.
    //
    if (BAL_OPERAND_TYPE_IMMEDIATE == metadata->operands[2].type)
    {
        const uint64_t imm12 = arm_registers[2];
        const uint64_t shift = 1 == sh ? 12 : 0;
        const uint64_t value = imm12 << shift;
        BAL_LOG_TRACE(context->logger,
                      "  Variant='Imm' Rd=%lu Rn=%lu Imm12=0x%lX Shift=%lu Value=0x%llX",
                      rd,
                      rn,
                      imm12,
                      shift,
                      value);
        const uint64_t rn_ssa_index_with_flag = get_or_create_ssa_index(context, rn);
        const uint64_t rn_ssa_index    = rn_ssa_index_with_flag & ~BAL_IS_CONSTANT_BIT_POSITION;
        const uint64_t imm12_ssa_index = intern_constant(context, value);

        if (BAL_UNLIKELY(context->status != BAL_SUCCESS))
        {
            return;
        }

        *context->ir_instruction_cursor = (bal_instruction_t)OPCODE_ADD << BAL_OPCODE_SHIFT_POSITION
                                          | rn_ssa_index << BAL_SOURCE1_SHIFT_POSITION
                                          | imm12_ssa_index << BAL_SOURCE2_SHIFT_POSITION;
        bal_bit_width_t bit_width = 0;

        if (BAL_OPERAND_TYPE_REGISTER_32 == metadata->operands[0].type)
        {
            bit_width = 32;
        }
        else if (BAL_OPERAND_TYPE_REGISTER_64 == metadata->operands[0].type)
        {
            bit_width = 64;
        }
        else
        {
            BAL_LOG_ERROR(context->logger, "Unknown register type for ADD (Imm) Rd register");
            context->status = BAL_ERROR_INCORRECT_REGISTER_TYPE;
            return;
        }

        *context->bit_width_cursor = bit_width;
        BAL_LOG_DEBUG(context->logger,
                      "  EMIT: v%u = ADD v%u, c%u (%u-bit)",
                      context->instruction_count,
                      rn_ssa_index,
                      imm12_ssa_index & ~BAL_IS_CONSTANT_BIT_POSITION,
                      bit_width);

        context->source_variables[rd].current_ssa_index = context->instruction_count;
        BAL_LOG_TRACE(
            context->logger, "  SSA UPDATE: X%lu -> v%lu", rd, context->instruction_count);
    }

    context->instruction_count++;
}

BAL_HOT static void
translate_const(bal_translation_context_t                *context,
                const bal_decoder_instruction_metadata_t *metadata,
                const uint32_t                           *arm_registers)
{
    const uint64_t rd    = arm_registers[0];
    const uint64_t imm16 = arm_registers[1];
    const uint64_t hw    = arm_registers[2];
    const uint64_t shift = hw * 16;

    const uint64_t mask = 0xFFFFFFFFFFFFFFFFULL;

    // Calculate the shifted immediate value.
    //
    uint64_t value = imm16 << shift & mask;

    // Check mnemonic 4th character: MOV[Z], MOV[N], MOV[K].
    //
    const char variant = metadata->name[3];

    BAL_LOG_TRACE(context->logger,
                  "  Variant='%c' Rd=%lu Imm=0x%lX Shift=%lu Mask=0x%llX",
                  variant,
                  rd,
                  imm16,
                  shift,
                  mask);

    if ('N' == variant)
    {
        value = ~value & mask;
        BAL_LOG_TRACE(context->logger, "  MOVN Inversion: New Value=0x%llX", value);
    }

    if ('K' == variant)
    {
        // MOVK:
        // mask = ~(0xFFFF << shift)
        // cleared_val = Old_Rd & mask
        // new_val = cleared_val + (imm << shift)

        uint64_t old_ssa;
        uint64_t old_ssa_with_flag;

        if (31 == rd)
        {
            BAL_LOG_TRACE(context->logger, "  MOVK Source is ZR. Interning 0.");
            old_ssa_with_flag = intern_constant(context, 0);
            old_ssa           = old_ssa_with_flag & ~BAL_IS_CONSTANT_BIT_POSITION;
        }
        else
        {
            old_ssa_with_flag = get_or_create_ssa_index(context, rd);
            old_ssa           = old_ssa_with_flag & ~BAL_IS_CONSTANT_BIT_POSITION;
            BAL_LOG_TRACE(context->logger, "  MOVK Source: Reg X%u -> SSA v%u", rd, old_ssa);
        }

        const uint64_t clear_mask = (~(0xFFFFULL << shift)) & mask;
        const uint64_t mask_index = intern_constant(context, clear_mask);

        if (BAL_UNLIKELY(context->status != BAL_SUCCESS))
        {
            return;
        }

        *context->ir_instruction_cursor = (bal_instruction_t)OPCODE_AND << BAL_OPCODE_SHIFT_POSITION
                                          | old_ssa_with_flag << BAL_SOURCE1_SHIFT_POSITION
                                          | mask_index << BAL_SOURCE2_SHIFT_POSITION;

        BAL_LOG_DEBUG(context->logger,
                      "  EMIT: v%u = AND v%u, c%u (Mask: 0x%llX)",
                      context->instruction_count,
                      old_ssa,
                      mask_index & ~BAL_IS_CONSTANT_BIT_POSITION,
                      clear_mask);

        const uint64_t cleared_ssa = context->instruction_count;

        // Remove unused variable warning from release builds.
        //
        (void)old_ssa;
        (void)cleared_ssa;

        // Advance cursor for the AND instruction.
        //
        context->ir_instruction_cursor++;
        context->bit_width_cursor++;
        context->instruction_count++;

        const uint64_t value_index = intern_constant(context, value);

        // Source 1 is the result of the AND instruction.
        //
        const uint64_t masked_ssa       = context->instruction_count - 1;
        *context->ir_instruction_cursor = (bal_instruction_t)OPCODE_ADD << BAL_OPCODE_SHIFT_POSITION
                                          | masked_ssa << BAL_SOURCE1_SHIFT_POSITION
                                          | (bal_instruction_t)value_index
                                                << BAL_SOURCE2_SHIFT_POSITION;

        BAL_LOG_DEBUG(context->logger,
                      "  EMIT: v%u = ADD v%u, c%u (Val: 0x%llX)",
                      context->instruction_count,
                      cleared_ssa,
                      value_index & ~BAL_IS_CONSTANT_BIT_POSITION,
                      value);
    }
    else
    {
        const uint64_t constant_index = intern_constant(context, value);

        if (BAL_UNLIKELY(context->status != BAL_SUCCESS))
        {
            return;
        }

        *context->ir_instruction_cursor
            = (bal_instruction_t)OPCODE_CONST << BAL_OPCODE_SHIFT_POSITION
              | (bal_instruction_t)constant_index << BAL_SOURCE1_SHIFT_POSITION;

        BAL_LOG_DEBUG(context->logger,
                      "  EMIT: v%u = CONST %u (0x%llX)",
                      context->instruction_count,
                      constant_index & ~BAL_IS_CONSTANT_BIT_POSITION,
                      value);
    }

    // Only update the SSA map is not writing to XZR/WZR.
    //
    if (rd != 31)
    {
        context->source_variables[rd].current_ssa_index = context->instruction_count;
        BAL_LOG_TRACE(
            (context)->logger, "  SSA UPDATE: X%u -> v%u", rd, context->instruction_count);
    }
    else
    {
        BAL_LOG_TRACE(context->logger, "    SSA NO-OP: Destination is XZR");
    }

    context->instruction_count++;
}

BAL_HOT static void
translate_return(bal_translation_context_t *context, const uint32_t *arm_registers)
{
    const uint64_t rn               = arm_registers[0];
    const uint64_t rn_ssa_index     = get_or_create_ssa_index(context, rn);
    *context->ir_instruction_cursor = (bal_instruction_t)OPCODE_RETURN << BAL_OPCODE_SHIFT_POSITION
                                      | rn_ssa_index << BAL_SOURCE1_SHIFT_POSITION;
    BAL_LOG_DEBUG(
        context->logger, "   EMIT: v%u = RET v%u", context->instruction_count, rn_ssa_index);
    context->instruction_count++;
}