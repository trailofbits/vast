// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
typedef int INT;
typedef INT INT2;

int fun1(int arg1, double arg2) {
    // CHECK: hl.cond : !hl.double {
    // CHECK: hl.cond.yield [[X:%[0-9]+]] : !hl.int
    // CHECK: } ? {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.double
    // CHECK: } : {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.double
    // CHECK: }
    int res = arg1 ? arg2 : arg1;
    return res;
}

INT2 fun2(INT2 arg1, INT arg2) {
    // CHECK: hl.cond : !hl.elaborated<!hl.typedef<@INT>> {
    // CHECK: hl.cond.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT2>>
    // CHECK: } ? {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT>>
    // CHECK: } : {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT2>>
    // CHECK: }
    INT res = arg1 ? arg2 : arg1;
    return res;
}

void* fun3(INT2 arg1, INT arg2) {
    // CHECK: hl.cond : !hl.ptr<!hl.elaborated<!hl.typedef<@INT>>> {
    // CHECK: hl.cond.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT2>>
    // CHECK: } ? {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.ptr<!hl.elaborated<!hl.typedef<@INT>>>
    // CHECK: } : {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.ptr<!hl.elaborated<!hl.typedef<@INT2>>>
    // CHECK: }
    void* res = arg1 ? &arg2 : &arg1;
    return res;
}
