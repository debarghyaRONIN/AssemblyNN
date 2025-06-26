.section .data

INPUT_SIZE:     .quad 2
HIDDEN_SIZE:    .quad 3
OUTPUT_SIZE:    .quad 1
LEARNING_RATE:  .double 0.1

# Weights: Input → Hidden (2x3)
w_ih:   .double 0.5, -0.2, 0.4
        .double 0.1, 0.4, -0.1

# Weights: Hidden → Output (3x1)
w_ho:   .double 0.2, -0.3, 0.6

# Biases
b_h:    .double 0.1, -0.1, 0.2
b_o:    .double 0.0

# Activations
input:      .double 0.0, 0.0
z_hidden:   .double 0.0, 0.0, 0.0
a_hidden:   .double 0.0, 0.0, 0.0
z_output:   .double 0.0
a_output:   .double 0.0

# Targets & Errors
target:     .double 0.0
error_out:  .double 0.0
error_hid:  .double 0.0, 0.0, 0.0

.section .rodata
const_0_25: .double 0.25
const_0_5:  .double 0.5
const_1_0:  .double 1.0
const_0_0:  .double 0.0
fmtstr:     .string "Output: %.6f\n"

.section .text
.global _start
.extern printf

_start:
    # Set input = [1.0, 0.0]; target = 1.0
    movsd const_1_0(%rip), %xmm0
    movsd %xmm0, input(%rip)
    movsd const_0_0(%rip), %xmm0
    movsd %xmm0, input+8(%rip)
    movsd const_1_0(%rip), %xmm0
    movsd %xmm0, target(%rip)

    # Forward + Backward pass
    call forward_pass
    call backward_pass

    # Print output
    movsd a_output(%rip), %xmm0
    lea fmtstr(%rip), %rdi
    mov $1, %rax        # xmm0 contains float arg
    call printf

    # Exit
    mov $60, %rax
    xor %rdi, %rdi
    syscall

# Forward propagation
forward_pass:
    push %rbp
    mov %rsp, %rbp
    xor %rcx, %rcx      # Hidden loop

.hidden_loop:
    cmp $3, %rcx
    jge .hidden_done

    movsd b_h(,%rcx,8), %xmm0
    xor %rdx, %rdx

.input_loop:
    cmp $2, %rdx
    jge .input_done

    movsd input(,%rdx,8), %xmm1
    mov %rdx, %rax
    imul $3, %rax
    add %rcx, %rax
    movsd w_ih(,%rax,8), %xmm2
    mulsd %xmm2, %xmm1
    addsd %xmm1, %xmm0

    inc %rdx
    jmp .input_loop

.input_done:
    movsd %xmm0, z_hidden(,%rcx,8)
    call sigmoid
    movsd %xmm0, a_hidden(,%rcx,8)
    inc %rcx
    jmp .hidden_loop

.hidden_done:
    movsd b_o(%rip), %xmm0
    xor %rcx, %rcx

.output_loop:
    cmp $3, %rcx
    jge .output_done

    movsd a_hidden(,%rcx,8), %xmm1
    movsd w_ho(,%rcx,8), %xmm2
    mulsd %xmm2, %xmm1
    addsd %xmm1, %xmm0

    inc %rcx
    jmp .output_loop

.output_done:
    movsd %xmm0, z_output(%rip)
    call sigmoid
    movsd %xmm0, a_output(%rip)

    pop %rbp
    ret

backward_pass:
    push %rbp
    mov %rsp, %rbp

    movsd target(%rip), %xmm0
    subsd a_output(%rip), %xmm0

    movsd a_output(%rip), %xmm1
    movsd const_1_0(%rip), %xmm2
    subsd %xmm1, %xmm2
    mulsd %xmm2, %xmm1
    mulsd %xmm1, %xmm0
    movsd %xmm0, error_out(%rip)

    xor %rcx, %rcx
.hidden_error_loop:
    cmp $3, %rcx
    jge .hidden_error_done

    movsd error_out(%rip), %xmm0
    mulsd w_ho(,%rcx,8), %xmm0

    movsd a_hidden(,%rcx,8), %xmm1
    movsd const_1_0(%rip), %xmm2
    subsd %xmm1, %xmm2
    mulsd %xmm2, %xmm1
    mulsd %xmm1, %xmm0
    movsd %xmm0, error_hid(,%rcx,8)

    inc %rcx
    jmp .hidden_error_loop

.hidden_error_done:
    call update_weights

    pop %rbp
    ret

# Update weights
update_weights:
    push %rbp
    mov %rsp, %rbp

    xor %rcx, %rcx
.update_ho_loop:
    cmp $3, %rcx
    jge .update_ho_done

    movsd LEARNING_RATE(%rip), %xmm0
    mulsd error_out(%rip), %xmm0
    mulsd a_hidden(,%rcx,8), %xmm0
    addsd w_ho(,%rcx,8), %xmm0
    movsd %xmm0, w_ho(,%rcx,8)

    inc %rcx
    jmp .update_ho_loop

.update_ho_done:
    movsd LEARNING_RATE(%rip), %xmm0
    mulsd error_out(%rip), %xmm0
    addsd b_o(%rip), %xmm0
    movsd %xmm0, b_o(%rip)

    xor %rcx, %rcx
.update_ih_outer:
    cmp $3, %rcx
    jge .update_ih_done

    xor %rdx, %rdx
.update_ih_inner:
    cmp $2, %rdx
    jge .update_ih_inner_done

    mov %rdx, %rax
    imul $3, %rax
    add %rcx, %rax

    movsd LEARNING_RATE(%rip), %xmm0
    mulsd error_hid(,%rcx,8), %xmm0
    mulsd input(,%rdx,8), %xmm0
    addsd w_ih(,%rax,8), %xmm0
    movsd %xmm0, w_ih(,%rax,8)

    inc %rdx
    jmp .update_ih_inner

.update_ih_inner_done:
    inc %rcx
    jmp .update_ih_outer

.update_ih_done:
    xor %rcx, %rcx
.update_bias_loop:
    cmp $3, %rcx
    jge .update_bias_done

    movsd LEARNING_RATE(%rip), %xmm0
    mulsd error_hid(,%rcx,8), %xmm0
    addsd b_h(,%rcx,8), %xmm0
    movsd %xmm0, b_h(,%rcx,8)

    inc %rcx
    jmp .update_bias_loop

.update_bias_done:
    pop %rbp
    ret

# Sigmoid Approximation: 0.5 + 0.25*x (clamped)
sigmoid:
    push %rbp
    mov %rsp, %rbp

    movsd const_0_25(%rip), %xmm1
    mulsd %xmm1, %xmm0
    movsd const_0_5(%rip), %xmm1
    addsd %xmm1, %xmm0

    movsd const_0_0(%rip), %xmm1
    maxsd %xmm1, %xmm0
    movsd const_1_0(%rip), %xmm1
    minsd %xmm1, %xmm0

    pop %rbp
    ret
