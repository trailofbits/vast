// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
typedef int INT;
typedef INT INT2;

int fun0(int arg1, int arg2) {
    // CHECK: hl.binary_cond : !hl.int {
    // CHECK: hl.value.yield {{%[0-9]+}} : [[type:!hl.int]]
    // CHECK: }, {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]: [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.cond.yield [[opaq]] : !hl.int
    // CHECK: } ? {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.value.yield [[opaq]] : !hl.int
    // CHECK: } : {
    // CHECK: hl.value.yield {{%[0-9]+}} : !hl.int
    // CHECK: }
    int res = arg1 ? : arg2;
    return res;
}

// CHECK: @fun1
int fun1(int arg1, double arg2) {
    // CHECK: hl.binary_cond : !hl.double {
    // CHECK: hl.value.yield {{%[0-9]+}} : [[type:!hl.int]]
    // CHECK: }, {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]: [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.cond.yield [[opaq]] : !hl.int
    // CHECK: } ? {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: [[X:%[0-9]+]] = hl.implicit_cast [[opaq]]
    // CHECK: hl.value.yield [[X]] : !hl.double
    // CHECK: } : {
    // CHECK: hl.value.yield {{%[0-9]+}} : !hl.double
    // CHECK: }
    int res = arg1 ? : arg2;
    return res;
}

// CHECK: @fun2
INT2 fun2(INT2 arg1, INT arg2) {
    // CHECK: hl.binary_cond : !hl.elaborated<!hl.typedef<"INT">> {
    // CHECK: hl.value.yield {{%[0-9]+}} : [[type:!hl.elaborated<!hl.typedef<"INT2">>]]
    // CHECK: }, {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]: [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.cond.yield [[opaq]] : !hl.elaborated<!hl.typedef<"INT2">>
    // CHECK: } ? {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]: [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.value.yield [[opaq]] : !hl.elaborated<!hl.typedef<"INT2">>
    // CHECK: } : {
    // CHECK: hl.value.yield {{%[0-9]+}} : !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: }
    INT res = arg1 ? : arg2;
    return res;
}

// CHECK: @fun3
void* fun3(INT2 arg1, INT arg2) {
    // CHECK: hl.binary_cond : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>> {
    // CHECK: hl.value.yield {{%[0-9]+}} : [[type:!hl.elaborated<!hl.typedef<"INT2">>]]
    // CHECK: }, {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]: [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.cond.yield [[opaq]] : !hl.elaborated<!hl.typedef<"INT2">>
    // CHECK: } ? {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]: [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: [[X:%[0-9]+]] = hl.implicit_cast [[opaq]]
    // CHECK: hl.value.yield [[X]] : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: } : {
    // CHECK: hl.value.yield {{%[0-9]+}} : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: }
    void* res = arg1 ? : &arg2;
    return res;
}
