// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

void bit_ops(int a, int b) {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.shl [[V2]], [[V4]]
    int shl = a << b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.shr [[V2]], [[V4]]
    int shr = a >> b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.xor [[V2]], [[V4]]
    int xor = a ^ b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.or [[V2]], [[V4]]
    int or  = a | b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.and [[V2]], [[V4]]
    int and = a & b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.land [[V2]], [[V4]]
    int land = a && b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.lor [[V2]], [[V4]]
    int lor  = a || b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.not [[V2]]
    int not = ~a;
}

void bit_assign_ops(int a, int b) {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.shl [[V4]] to [[V1]]
    a <<= b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.shr [[V4]] to [[V1]]
    a >>= b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.xor [[V4]] to [[V1]]
    a ^= b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.or [[V4]] to [[V1]]
    a |= b;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @a
    // CHECK: [[V2:%[0-9]+]] = hl.declref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.and [[V4]] to [[V1]]
    a &= b;
}
