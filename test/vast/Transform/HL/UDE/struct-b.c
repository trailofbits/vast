// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK: hl.struct @used
struct used {};

// CHECK-NOT: hl.struct @unused
struct unused {
    struct used d;
};

int main() {
    struct used u;
}
