// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

#define CHAR_BIT 8

// CHECK: sign1(%arg0: !hl.int) -> !hl.int
int sign1(int v) {
    // CHECK: [[V0:%[0-9]+]] = hl.expr : !hl.int
    // CHECK:   [[V1:%[0-9]+]] = hl.declref @v : !hl.int
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.int -> !hl.int
    // CHECK:   [[V3:%[0-9]+]] = hl.constant.int 0 : !hl.int
    // CHECK:   [[V4:%[0-9]+]] = hl.cmp slt !hl.int [[V2]], [[V3]] -> !hl.int
    // CHECK: hl.minus [[V0]] : !hl.int
    return -(v < 0);
}

int sign2(int v) {
    // CHECK: hl.expr : !hl.int<unsigned>
    // CHECK:  hl.expr : !hl.int
    // CHECK:   hl.declref @v
    // CHECK:   hl.cstyle_cast [[X:%[0-9]+]] NoOp : !hl.int -> !hl.int
    // CHECK:  hl.cstyle_cast [[Y:%[0-9]+]] IntegralCast : !hl.int -> !hl.int<unsigned>
    // CHECK:  hl.expr : !hl.long<unsigned>
    // CHECK:   hl.sizeof.type !hl.int -> !hl.long<unsigned>
    // CHECK:   hl.constant.int 8 : !hl.int
    // CHECK:   hl.mul
    // CHECK:   hl.constant.int 1
    // CHECK:   hl.sub
    // CHECK:  hl.bin.shr
    // CHECK: hl.minus
    return -(int)((unsigned int)((int)v) >> (sizeof(int) * CHAR_BIT - 1));
}

int sign3(unsigned long v) {
    // CHECK: [[V0:%[0-9]+]] = hl.declref @v : !hl.long<unsigned>
    // CHECK: [[V1:%[0-9]+]] = hl.implicit_cast [[V0]]
    // CHECK: [[V2:%[0-9]+]] = hl.expr : !hl.long<unsigned>
    // CHECK:  hl.sizeof.type !hl.int -> !hl.long<unsigned>
    // CHECK:  hl.constant.int 8 : !hl.int
    // CHECK:  hl.mul
    // CHECK:  hl.constant.int 1 : !hl.int
    // CHECK:  hl.sub
    // CHECK: hl.bin.shr [[V1]], [[V2]]
    return v >> (sizeof(int) * CHAR_BIT - 1);
}
