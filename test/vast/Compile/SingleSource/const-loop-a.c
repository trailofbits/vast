// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

int loop()
{
    const int N = 100;
    int sum = 0;
    for (int i = 1; i <= N; ++i)
    {
        sum += i;
    }
    return sum;
}

// HL: hl.func @loop {{.*}} () -> !hl.int

// MLIR: llvm.func @loop() -> i32

// LLVM: define i32 @loop()