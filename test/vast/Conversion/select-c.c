// RUN: %vast-front -vast-emit-mlir-after=vast-irs-to-llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=I_LLVM

// RUN: %vast-front -vast-emit-mlir-after=vast-core-to-llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=C_LLVM

int main(int argc, char** argv)
{
    // I_LLVM: [[V6:%[0-9]+]] = core.lazy.op {
    // I_LLVM:   [[V13:%[0-9]+]] = core.lazy.op {
    // I_LLVM:     [[V16:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
    // I_LLVM:     hl.value.yield [[V16]] : i32
    // I_LLVM:   } : i32
    // I_LLVM:   [[V14:%[0-9]+]] = core.lazy.op {
    // I_LLVM:     [[V26:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
    // I_LLVM:     hl.value.yield [[V26]] : i32
    // I_LLVM:   } : i32
    // I_LLVM:   [[V15:%[0-9]+]] = core.select {{.*}}, [[V13]], [[V14]] : (i32, i32, i32) -> i32
    // I_LLVM:   hl.value.yield [[V15]] : i32
    // I_LLVM: } : i32
    // I_LLVM: [[V7:%[0-9]+]] = core.lazy.op {
    // I_LLVM:   [[V9:%[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
    // I_LLVM:   hl.value.yield [[V9]] : i32
    // I_LLVM: } : i32
    // I_LLVM: [[V8:%[0-9]+]] = core.select {{.*}}, [[V6]], [[V7]] : (i32, i32, i32) -> i32
    // I_LLVM: llvm.return [[V8]] : i32

    // C_LLVM:  llvm.cond_br {{.*}}, ^bb1, ^bb5
    // C_LLVM: ^bb1:  // pred: ^bb0
    // C_LLVM:  {{.*}} = llvm.mlir.constant(3 : i32) : i32
    // C_LLVM:  llvm.cond_br %13, ^bb2, ^bb3
    // C_LLVM: ^bb2:  // pred: ^bb1
    // C_LLVM:  [[V14:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
    // C_LLVM:  llvm.br ^bb4([[V14]] : i32)
    // C_LLVM: ^bb3:  // pred: ^bb1
    // C_LLVM:  [[V15:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
    // C_LLVM:  llvm.br ^bb4([[V15]] : i32)
    // C_LLVM: ^bb4([[V16:%[0-9]+]]: i32):  // 2 preds: ^bb2, ^bb3
    // C_LLVM:  llvm.br ^bb6([[V16]] : i32)
    // C_LLVM: ^bb5:  // pred: ^bb0
    // C_LLVM:  [[V17:%[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
    // C_LLVM:  llvm.br ^bb6([[V17]] : i32)
    // C_LLVM: ^bb6([[V18:%[0-9]+]]: i32):  // 2 preds: ^bb4, ^bb5
    // C_LLVM:  llvm.return [[V18]] : i32


    return (argc >= 3) ? ( ( argc == 3) ? 0 : 1 )
                       : 2;
}
