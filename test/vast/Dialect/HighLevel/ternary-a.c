// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
typedef int INT;
typedef INT INT2;

int fun1(int arg1, double arg2) {
    // CHECK: hl.cond {
    // CHECK: hl.cond.yield [[X:%[0-9]+]] : !hl.int
    // CHECK: } ? {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.double
    // CHECK: } : {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.double
    // CHECK: } : !hl.double
    int res = arg1 ? arg2 : arg1;
    return res;
}

INT2 fun2(INT2 arg1, INT arg2) {
    // CHECK: hl.cond {
    // CHECK: hl.cond.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<"INT2">>
    // CHECK: } ? {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: } : {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<"INT2">>
    // CHECK: } : !hl.elaborated<!hl.typedef<"INT">>
    INT res = arg1 ? arg2 : arg1;
    return res;
}

void* fun3(INT2 arg1, INT arg2) {
    // CHECK: hl.cond {
    // CHECK: hl.cond.yield [[X:%[0-9]+]] : !hl.elaborated<!hl.typedef<"INT2">>
    // CHECK: } ? {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: } : {
    // CHECK: hl.value.yield [[X:%[0-9]+]] : !hl.ptr<!hl.elaborated<!hl.typedef<"INT2">>>
    // CHECK: } : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    void* res = arg1 ? &arg2 : &arg1;
    return res;
}
