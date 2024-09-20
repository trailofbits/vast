// RUN: %vast-cc1 -vast-emit-mlir-after=vast-lower-value-categories %s -o - | %file-check %s

enum E { A, B, C };

void enum_resolve() {
    // CHECK: [[E:%[0-9]+]] = ll.alloca : !hl.ptr<ui32>
    // CHECK: [[V:%[0-9]+]] = hl.const #core.integer<1>
    // CHECK: [[C:%[0-9]+]] = hl.implicit_cast [[V]] IntegralCast
    // CHECK: ll.store [[E]], [[C]]
    enum E e = B;
}

void enum_cast() {
    // CHECK: [[E:%[0-9]+]] = ll.alloca : !hl.ptr<ui32>
    // CHECK: [[V:%[0-9]+]] = hl.const #core.integer<2>
    // CHECK: [[C:%[0-9]+]] = hl.cstyle_cast [[V]] IntegralCast
    // CHECK: ll.store [[E]], [[C]]
    enum E e = (enum E)2;
}

void enum_in_scope() {
    // CHECK: [[E:%[0-9]+]] = ll.alloca : !hl.ptr<ui32>
    enum E e;
    {
        // CHECK: [[V:%[0-9]+]] = hl.const #core.integer<0>
        // CHECK: [[C:%[0-9]+]] = hl.implicit_cast [[V]] IntegralCast
        // CHECK: ll.store [[E]], [[C]]
        e = A;
    }
}

typedef enum E e_t;

void enum_typedef() {
    // CHECK: [[E:%[0-9]+]] = ll.alloca : !hl.ptr<ui32>
    // CHECK: [[V:%[0-9]+]] = hl.const #core.integer<0>
    // CHECK: [[C:%[0-9]+]] = hl.implicit_cast [[V]] IntegralCast
    // CHECK: ll.store [[E]], [[C]]
    e_t e = A;
}


void enum_param(enum E e) {
    // CHECK: [[E:%[0-9]+]] = ll.alloca : !hl.ptr<ui32>
    // CHECK: ll.store [[E]], %arg0
}

enum E return_enum() {
    // CHECK: [[V:%[0-9]+]] = hl.const #core.integer<1>
    // CHECK: [[C:%[0-9]+]] = hl.implicit_cast [[V]] IntegralCast
    // CHECK: ll.return [[C]]
    return B;
}
