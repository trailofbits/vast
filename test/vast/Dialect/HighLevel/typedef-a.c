// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.typedef "INT" : !hl.int
// CHECK: hl.typedef "SHORT" : !hl.short
typedef int INT;
typedef short SHORT;
// CHECK: hl.func @arithemtic_int_short {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>>)
void arithemtic_int_short(INT a, SHORT b)
{
    INT c;
    // CHECK: [[C:%[0-9]+]] = hl.var @c : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : (!hl.elaborated<!hl.typedef<"INT">>, !hl.elaborated<!hl.typedef<"INT">>) -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.elaborated<!hl.typedef<"INT">>, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = a + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V2]], [[V5]] : (!hl.elaborated<!hl.typedef<"INT">>, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V5]], [[V4]] : (!hl.int, !hl.elaborated<!hl.typedef<"INT">>) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = b + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V2]], [[V5]] : (!hl.elaborated<!hl.typedef<"INT">>, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = a - b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V5]], [[V4]] : (!hl.int, !hl.elaborated<!hl.typedef<"INT">>) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = b - a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V2]], [[V5]] : (!hl.elaborated<!hl.typedef<"INT">>, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = a * b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V5]], [[V4]] : (!hl.int, !hl.elaborated<!hl.typedef<"INT">>) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = b * a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V2]], [[V5]] : (!hl.elaborated<!hl.typedef<"INT">>, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = a / b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V5]], [[V4]] : (!hl.int, !hl.elaborated<!hl.typedef<"INT">>) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = b / a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V2]], [[V5]] : (!hl.elaborated<!hl.typedef<"INT">>, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = a % b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"SHORT">>> -> !hl.elaborated<!hl.typedef<"SHORT">>
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.elaborated<!hl.typedef<"SHORT">> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V5]], [[V4]] : (!hl.int, !hl.elaborated<!hl.typedef<"INT">>) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int, !hl.lvalue<!hl.elaborated<!hl.typedef<"INT">>> -> !hl.elaborated<!hl.typedef<"INT">>
    c = b % a;
}

void arithemtic_pointer_long(INT* a, long b)
{
    INT* c;
    // CHECK: [[C:%[0-9]+]] = hl.var @c : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>>
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>> -> !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : (!hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>, !hl.long) -> !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>> -> !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : (!hl.long, !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>) -> !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    c = b + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref [[C]]
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>> -> !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: [[V3:%[0-9]+]] = hl.ref [[A2]]
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V5:%[0-9]+]] = hl.sub [[V2]], [[V4]] : (!hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>, !hl.long) -> !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.ptr<!hl.elaborated<!hl.typedef<"INT">>>
    c = a - b;
}
