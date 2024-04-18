// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

int conditional()
{
    int x = 42;
    int y = 0;
    if (x > 1)
        y = 1;
    else
        y = 2;
    return y;
}

// HL: hl.func @conditional {{.*}} () -> !hl.int

// MLIR: llvm.func @conditional() -> i32

// LLVM: define i32 @conditional()