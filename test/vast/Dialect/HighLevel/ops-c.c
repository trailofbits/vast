// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void bit_ops(int a, int b) {
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.shl [[V2]], [[V4]]
    int shl = a << b;
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.ashr [[V2]], [[V4]]
    int shr = a >> b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.xor [[V2]], [[V4]]
    int xor = a ^ b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.or [[V2]], [[V4]]
    int or  = a | b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.bin.and [[V2]], [[V4]]
    int and = a & b;

    // CHECK: hl.bin.land {
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[A:%[0-9]+]] : !hl.int
    // CHECK: }, {
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.value.yield [[B:%[0-9]+]] : !hl.int
    // CHECK: } : !hl.int
    int land = a && b;

    // CHECK: hl.bin.lor {
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[A:%[0-9]+]] : !hl.int
    // CHECK: }, {
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]]
    // CHECK: hl.value.yield [[B:%[0-9]+]] : !hl.int
    // CHECK: } : !hl.int
    int lor  = a || b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.not [[V2]]
    int not = ~a;
}

void bit_assign_ops(int a, int b) {
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.shl [[V4]] to [[V1]]
    a <<= b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.ashr [[V4]] to [[V1]]
    a >>= b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.xor [[V4]] to [[V1]]
    a ^= b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.or [[V4]] to [[V1]]
    a |= b;

    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.assign.bin.and [[V4]] to [[V1]]
    a &= b;
}
