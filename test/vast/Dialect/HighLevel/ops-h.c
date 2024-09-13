// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @arithemtic_int_short {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.short>)
void arithemtic_int_short(int a, short b)
{
    int c;
    // CHECK: hl.var @c : !hl.lvalue<!hl.int>
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V2]], [[V5]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V5]], [[V4]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = b + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V2]], [[V5]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = a - b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V5]], [[V4]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = b - a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V2]], [[V5]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = a * b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V5]], [[V4]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = b * a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V2]], [[V5]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = a / b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V5]], [[V4]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = b / a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V2]], [[V5]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = a % b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V5]], [[V4]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.int
    c = b % a;
}

void arithemtic_int_long(int a, long b)
{
    long c;
    // CHECK: hl.var @c : !hl.lvalue<!hl.long>
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a - b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b - a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a * b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b * a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a / b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b / a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a % b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.int -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b % a;
}

void arithemtic_short_long(short a, long b)
{
    long c;
    // CHECK: hl.var @c : !hl.lvalue<!hl.long>
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a - b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b - a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a * b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.mul [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b * a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a / b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.sdiv [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b / a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V4:%[0-9]+]] = hl.ref @b
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V3]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = a % b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.long
    // CHECK: [[V6:%[0-9]+]] = hl.srem [[V2]], [[V5]] : (!hl.long, !hl.long) -> !hl.long
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.long
    c = b % a;
}

void arithemtic_pointer_short(int* a, short b)
{
    int* c;
    // CHECK: hl.var @c : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V2]], [[V5]] : (!hl.ptr<!hl.int>, !hl.int) -> !hl.ptr<!hl.int>
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.ptr<!hl.int>
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V4:%[0-9]+]] = hl.ref @a
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V6:%[0-9]+]] = hl.add [[V3]], [[V5]] : (!hl.int, !hl.ptr<!hl.int>) -> !hl.ptr<!hl.int>
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.ptr<!hl.int>
    c = b + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.short> -> !hl.short
    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V4]] IntegralCast : !hl.short -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.sub [[V2]], [[V5]] : (!hl.ptr<!hl.int>, !hl.int) -> !hl.ptr<!hl.int>
    // CHECK: hl.assign [[V6]] to [[CR]] : !hl.ptr<!hl.int>
    c = a - b;
}

void arithemtic_pointer_long(int* a, long b)
{
    int* c;
    // CHECK: hl.var @c : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : (!hl.ptr<!hl.int>, !hl.long) -> !hl.ptr<!hl.int>
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.ptr<!hl.int>
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @b
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V3:%[0-9]+]] = hl.ref @a
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : (!hl.long, !hl.ptr<!hl.int>) -> !hl.ptr<!hl.int>
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.ptr<!hl.int>
    c = b + a;
    // CHECK: [[CR:%[0-9]+]] = hl.ref @c
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.long> -> !hl.long
    // CHECK: [[V5:%[0-9]+]] = hl.sub [[V2]], [[V4]] : (!hl.ptr<!hl.int>, !hl.long) -> !hl.ptr<!hl.int>
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.ptr<!hl.int>
    c = a - b;
}
