// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: func @arithemtic_signed([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.void
void arithemtic_signed(int a, int b)
{
    int c;
    // CHECK: [[C:%[0-9]+]] = hl.var "c" : !hl.lvalue<!hl.int>
    // CHECK: [[CR:%[0-9]+]] = hl.decl.ref [[C]] : !hl.lvalue<!hl.int>
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]]
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.int
    c = a + b;
    // CHECK: [[CR:%[0-9]+]] = hl.decl.ref [[C]] : !hl.lvalue<!hl.int>
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.sub [[V2]], [[V4]]
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.int
    c = a - b;
    // CHECK: [[CR:%[0-9]+]] = hl.decl.ref [[C]] : !hl.lvalue<!hl.int>
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.mul [[V2]], [[V4]]
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.int
    c = a * b;
    // CHECK: [[CR:%[0-9]+]] = hl.decl.ref [[C]] : !hl.lvalue<!hl.int>
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.sdiv [[V2]], [[V4]]
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.int
    c = a / b;
    // CHECK: [[CR:%[0-9]+]] = hl.decl.ref [[C]] : !hl.lvalue<!hl.int>
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.srem [[V2]], [[V4]]
    // CHECK: hl.assign [[V5]] to [[CR]] : !hl.int
    c = a % b;
}

void arithemtic_unsigned(unsigned a, unsigned b)
{
    unsigned int c;
    // CHECK: hl.udiv
    c = a / b;
    // CHECK: hl.urem
    c = a % b;
}

// CHECK: func @assign_signed([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.void
void assign_signed(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.assign.add [[V3]] to [[V1]] : !hl.int
    a += b;
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.assign.sub [[V3]] to [[V1]] : !hl.int
    a -= b;
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.assign.mul [[V3]] to [[V1]] : !hl.int
    a *= b;
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.assign.sdiv [[V3]] to [[V1]] : !hl.int
    a /= b;
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.assign.srem [[V3]] to [[V1]] : !hl.int
    a %= b;
}

// CHECK: func @assign_unsigned([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int<unsigned>>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.int<unsigned>>) -> !hl.void {
void assign_unsigned(unsigned a, unsigned b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int<unsigned>>
    // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int<unsigned>>
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int<unsigned>> -> !hl.int<unsigned>
    // CHECK: hl.assign.udiv [[V3]] to [[V1]] : !hl.int<unsigned>
    a /= b;
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int<unsigned>>
    // CHECK: [[V2:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int<unsigned>>
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int<unsigned>> -> !hl.int<unsigned>
    // CHECK: hl.assign.urem [[V3]] to [[V1]] : !hl.int<unsigned>
    a %= b;
}
