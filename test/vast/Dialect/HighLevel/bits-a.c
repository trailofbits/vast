// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#define CHAR_BIT 8

// CHECK: @sign1 {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int sign1(int v) {
    // CHECK: [[V0:%[0-9]+]] = hl.expr : !hl.int
    // CHECK:   [[V1:%[0-9]+]] = hl.ref [[A1]]
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:   [[V3:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
    // CHECK:   [[V4:%[0-9]+]] = hl.cmp slt [[V2]], [[V3]] : !hl.int, !hl.int -> !hl.int
    // CHECK: hl.minus [[V0]] : !hl.int
    return -(v < 0);
}

// CHECK: @sign2 {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int sign2(int v) {
    // CHECK: hl.expr : !hl.int< unsigned >
    // CHECK:  hl.expr : !hl.int
    // CHECK:   hl.ref [[A1]]
    // CHECK:   hl.cstyle_cast [[X:%[0-9]+]] NoOp : !hl.int -> !hl.int
    // CHECK:  hl.cstyle_cast [[Y:%[0-9]+]] IntegralCast : !hl.int -> !hl.int< unsigned >
    // CHECK:  hl.expr : !hl.long< unsigned >
    // CHECK:   hl.sizeof.type !hl.int -> !hl.long< unsigned >
    // CHECK:   hl.const #core.integer<8> : !hl.int
    // CHECK:   hl.mul
    // CHECK:   hl.const #core.integer<1> : !hl.int
    // CHECK:   hl.sub
    // CHECK:  hl.bin.lshr
    // CHECK: hl.minus
    return -(int)((unsigned int)((int)v) >> (sizeof(int) * CHAR_BIT - 1));
}

// CHECK: @sign3 {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.long< unsigned >>) -> !hl.int
int sign3(unsigned long v) {
    // CHECK: [[V0:%[0-9]+]] = hl.ref [[A1]]
    // CHECK: [[V1:%[0-9]+]] = hl.implicit_cast [[V0]]
    // CHECK: [[V2:%[0-9]+]] = hl.expr : !hl.long< unsigned >
    // CHECK:  hl.sizeof.type !hl.int -> !hl.long< unsigned >
    // CHECK:  hl.const #core.integer<8> : !hl.int
    // CHECK:  hl.mul
    // CHECK:  hl.const #core.integer<1> : !hl.int
    // CHECK:  hl.sub
    // CHECK: hl.bin.lshr [[V1]], [[V2]]
    return v >> (sizeof(int) * CHAR_BIT - 1);
}
