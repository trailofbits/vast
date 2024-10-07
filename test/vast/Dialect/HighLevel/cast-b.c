// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.typedef @function_type : !core.fn<(!hl.lvalue<!hl.int>, !hl.lvalue<!hl.int>) -> (!hl.int)>
typedef int function_type(int a, int b);

// CHECK: hl.var @p, <external> : !hl.lvalue<!hl.array<2, !hl.ptr<!hl.elaborated<!hl.typedef<@function_type>>>>>
function_type *p[2];

int test_func(void) {
    // CHECK: [[F:%[0-9]+]] = hl.implicit_cast [[P:%[0-9]+]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.typedef<@function_type>>>> -> !hl.ptr<!hl.elaborated<!hl.typedef<@function_type>>>
    // CHECK: [[A:%[0-9]+]] = hl.const #core.integer<15> : !hl.int
    // CHECK: [[B:%[0-9]+]] = hl.const #core.integer<20> : !hl.int
    // CHECK: hl.indirect_call [[F]] : !hl.ptr<!hl.elaborated<!hl.typedef<@function_type>>>([[A]], [[B]]) : (!hl.int, !hl.int) -> !hl.int
    return p[1](15, 20);
}
