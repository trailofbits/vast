// RUN: vast-cc --ccopts -xc --ccopts -std=c89 --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --ccopts -std=c89 --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    // CHECK: hl.call @implicit_function() : () -> !hl.int
    implicit_function();

    int x, y;
    // CHECK: hl.call @implicit_function_with_args([[A1:%[0-9]+]], [[A2:%[0-9]+]]) : (!hl.int, !hl.int) -> !hl.int
    implicit_function_with_args(x, y);
}
