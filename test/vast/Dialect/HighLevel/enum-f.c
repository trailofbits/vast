// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.enum @kobj_ns_type
enum kobj_ns_type;
// CHECK: hl.struct
struct x {enum kobj_ns_type *type; int a;};
// CHECK: hl.enum @kobj_ns_type : !hl.int< unsigned >
// CHECK:   hl.enum.const @KOBJ_NS_TYPE_NONE
// CHECK:   hl.enum.const @KOBJ_NS_TYPE_NET
// CHECK:   hl.enum.const @KOBJ_NS_TYPES
enum kobj_ns_type {
 KOBJ_NS_TYPE_NONE = 0,
 KOBJ_NS_TYPE_NET,
 KOBJ_NS_TYPES
};


int main(void) {
    enum kobj_ns_type type = KOBJ_NS_TYPE_NONE;
return 0;
}
