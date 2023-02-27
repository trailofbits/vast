// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.typedef "INT" : !hl.int
// CHECK: hl.typedef "INT2" : !hl.long
typedef int INT;
typedef long INT2;
// CHECK: hl.func external @fun ([[A0:%arg[0-9]+]]: !hl.lvalue<!hl.typedef<"INT">>, [[A1:%arg[0-9]+]]: !hl.lvalue<!hl.typedef<"INT2">>) -> !hl.typedef<"INT">
INT fun(INT a, INT2 b) {
    // CHECK: [[V0:%[0-9]+]] = hl.ref [[A0]] : !hl.lvalue<!hl.typedef<"INT">>
    // CHECK: [[V1:%[0-9]+]] = hl.post.inc [[V0]] : !hl.lvalue<!hl.typedef<"INT">> -> !hl.typedef<"INT">
    a++;
    // CHECK: [[LOR:%[0-9]+]] = hl.bin.lor {
        // CHECK: [[EQ:%[0-9]+]] = hl.cmp eq [[X:%[0-9]+]], [[Y:%[0-9]+]] : !hl.long, !hl.typedef<"INT2"> -> !hl.int
        // CHECK: hl.value.yield [[EQ]] : !hl.int
    // CHECK: }, {
        // CHECK: [[NOT:%[0-9]+]] = hl.lnot [[X:%[0-9]+]] : !hl.typedef<"INT2">
        // CHECK: hl.value.yield [[NOT]]
        // CHECK: } : !hl.int
    // CHECK! hl.cond.yield [[LOR]] : hl.int
    if(a == b || !b)
    // CHECK: [[V0:%[0-9]+]] = hl.ref [[A1]] : !hl.lvalue<!hl.typedef<"INT2">>
    // CHECK: [[V1:%[0-9]+]] = hl.post.inc [[V0]] : !hl.lvalue<!hl.typedef<"INT2">> -> !hl.typedef<"INT2">
        b++;
    // CHECK: [[C:%[0-9]+]] = hl.var "c" : !hl.lvalue<!hl.int> = {
        // CHECK: [[SHL:%[0-9]+]] = hl.bin.shl [[X:%[0-9]+]], [[Y:%[0-9]+]] : (!hl.typedef<"INT2">, !hl.int) -> !hl.typedef<"INT2">
    // CHECK: }
    int c = b<<1;
    // CHECK: [[SHR:%[0-9]+]] = hl.bin.shr [[X:%[0-9]+]], [[Y:%[0-9]+]] : (!hl.typedef<"INT">, !hl.int) -> !hl.typedef<"INT">
    c = a>>1;
    // CHECK: [[PTR:%[0-9]+]] = hl.var "ptr" : !hl.lvalue<!hl.ptr<!hl.typedef<"INT">>> = {
        // CHECK: hl.addressof [[X:%[0-9]+]] : !hl.lvalue<!hl.typedef<"INT">> -> !hl.ptr<!hl.typedef<"INT">>
    // CHECK: }
    INT *ptr = &a;
    // CHECK: hl.pre.inc [[X:%[0-9]+]] : !hl.lvalue<!hl.typedef<"INT">> -> !hl.typedef<"INT">
    ++(*ptr);
    // CHECK: hl.pre.inc [[X:%[0-9]+]] : !hl.lvalue<!hl.ptr<!hl.typedef<"INT">>> -> !hl.ptr<!hl.typedef<"INT">>
    ++ptr;
    INT d = c;
    // CHECK: hl.if {
        // CHECK: hl.cmp slt [[X:%[0-9]+]], [[Y:%[0-9]+]] : !hl.typedef<"INT">, !hl.int -> !hl.int
    // CHECK: } then {
        // CHECK: hl.minus [[X:%[0-9]+]] : !hl.typedef<"INT2">
    // CHECK: }
    if (d < c)
        d = -b;
    // CHECK: hl.implicit_cast [[X:%[0-9]+]] IntegralCast : !hl.typedef<"INT2"> -> !hl.typedef<"INT">
    return b;
}

