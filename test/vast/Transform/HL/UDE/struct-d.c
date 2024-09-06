// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.struct @in_unused
struct in_unused {};

// CHECK-NOT: hl.struct @dependent_unused
struct dependent_unused {
    struct in_unused d;
};

int main() {}
