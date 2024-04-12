// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM
// REQUIRES: type-conversion

int factorial(int n)
{
    if (n <= 1)
    {
        return 1;
    }
    else
    {
        return n * factorial(n - 1);
    }
}

int main()
{
    int n = 5;
    int result = factorial(n);
    return result;
}

// HL: hl.func @factorial (!hl.int) -> !hl.int

// MLIR: llvm.func @conditional(i32) -> i32

// LLVM: define i32 @conditional(i32)