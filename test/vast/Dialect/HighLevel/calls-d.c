// RUN: %vast-cc1 -vast-emit-mlir=hl -std=c89 %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl -std=c89 %s -o %t && %vast-opt %t | diff -B %t -

int main()
{
    // CHECK: hl.call @implicit_function() : () -> !hl.int
    implicit_function();

    int x, y;
    // CHECK: hl.call @implicit_function_with_args([[A1:%[0-9]+]], [[A2:%[0-9]+]]) : (!hl.int, !hl.int) -> !hl.int
    implicit_function_with_args(x, y);
}
