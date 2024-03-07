// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.struct "unused_recursive"
struct unused_recursive {
    struct unused_recursive *v;
};

// CHECK-NOT: hl.typedecl "unused_typedefed_recursive"
struct unused_typedefed_recursive;

// CHECK-NOT: hl.struct "unused_recursive_t"
typedef struct unused_typedefed_recursive unused_recursive_t;

// CHECK-NOT: hl.struct "unused_typedefed_recursive"
struct unused_typedefed_recursive {
    unused_recursive_t *v;
};