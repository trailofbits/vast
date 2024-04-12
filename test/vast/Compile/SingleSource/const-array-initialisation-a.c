// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -S %s -o %t.s

int array_sum()
{
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;

    for (int i = 0; i < 5; ++i)
    {
        sum += arr[i];
    }

    return sum;
}

// HL: hl.func @array_sum () -> !hl.int

// MLIR: llvm.func @array_sum() -> i32

// LLVM: define i32 @array_sum()