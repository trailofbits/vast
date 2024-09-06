// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.struct @unused
struct unused {};

// CHECK: hl.struct @used
struct used {};

// CHECK-NOT: hl.func @inlined
__attribute__((always_inline)) void inlined() {
    struct unused un;
    struct used u;
}

// CHECK: hl.func @not_inlined
void not_inlined() {
    struct used u;
}
