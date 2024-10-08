// RUN: %vast-front -vast-emit-mlir=llvm %s -o - | %file-check %s
#include <stdio.h>

int main(int argc, char **argv)
{
    // CHECK: llvm.call @printf({{.*}}, {{.*}}) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    printf("argc: %i\n", argc);
}
