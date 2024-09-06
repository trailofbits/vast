// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.struct @used
struct used {};

// CHECK-NOT: hl.union @unused_union
union unused_union
{
    struct used s;
};
