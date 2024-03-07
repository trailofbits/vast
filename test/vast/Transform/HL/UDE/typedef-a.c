// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK: hl.struct "used"
struct used {};

// CHECK-NOT: hl.typedef "unused_typedef"
typedef struct used unused_typedef;

int main() {
    struct used u;
}