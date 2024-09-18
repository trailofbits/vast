// RUN: %vast-front -vast-emit-mlir=llvm %s -o - | %file-check %s

#include <stdint.h>

int main() {
    int64_t x = 5;
    uint64_t y = 5;
    // CHECK: llvm.shl {{.*}} : i64
    int64_t u = x << 3;
    // CHECK: llvm.ashr {{.*}} : i64
    int64_t v = x >> 3;
    // CHECK: llvm.lshr {{.*}} : i64
    uint64_t w = y >> 3;
    return 0;
}
