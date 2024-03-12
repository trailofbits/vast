// RUN: %vast-front -vast-emit-mlir=hl %s -o - | vast-opt --vast-hl-to-hl-builtin | %file-check %s

// REQUIRES: addrof.builtin
#include <cstdio>

int main() {
    auto f = &printf;
    // CHECK: hlbi.printf
    f("test");
}
