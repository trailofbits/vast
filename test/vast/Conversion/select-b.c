// RUN: %vast-front -vast-emit-mlir-after=vast-irs-to-llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=I_LLVM

// RUN: %vast-front -vast-emit-mlir-after=vast-core-to-llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=C_LLVM

void foo() {}

int main(int argc, char** argv)
{
    // I_LLVM: [[V6:%[0-9]+]] = core.lazy.op {
    // I_LLVM:   llvm.call @foo() : () -> ()
    // I_LLVM:   [[V11:%[0-9]+]] = llvm.mlir.zero : !llvm.void
    // I_LLVM:   hl.value.yield [[V11]] : !llvm.void
    // I_LLVM: } : !llvm.void
    // I_LLVM: [[V7:%[0-9]+]] = core.lazy.op {
    // I_LLVM:   [[V12:%[0-9]+]] = llvm.mlir.zero : !llvm.void
    // I_LLVM:   hl.value.yield [[V12]] : !llvm.void
    // I_LLVM: } : !llvm.void
    // I_LLVM: {{.*}} = core.select {{.*}}, [[V6]], [[V7]] : (i32, !llvm.void, !llvm.void) -> !llvm.void

    // C_LLVM:   llvm.cond_br {{.*}}, ^bb1, ^bb2
    // C_LLVM: ^bb1:  // pred: ^bb0
    // C_LLVM:   llvm.call @foo() : () -> ()
    // C_LLVM:   llvm.br ^bb3
    // C_LLVM: ^bb2:  // pred: ^bb0
    // C_LLVM:   llvm.br ^bb3
    // C_LLVM: ^bb3:  // 2 preds: ^bb1, ^bb2
    (argc >= 3) ? foo() : (void)0;
}
