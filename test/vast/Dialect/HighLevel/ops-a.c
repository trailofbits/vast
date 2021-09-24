// RUN: vast-cc --from-source %s | FileCheck %s

int add1(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a ): !hl.int
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast( [[V1]] ): !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.declref( @b ): !hl.int
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast( [[V3]] ): !hl.int
    // CHECK: hl.add [[V2]], [[V4]] : !hl.int
    return a + b;
}

int add2(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a ): !hl.int
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast( [[V1]] ): !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.declref( @b ): !hl.int
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast( [[V3]] ): !hl.int
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.var( r, [[V5]] ): !hl.int
    int r = a + b;

    // CHECK: [[V7:%[0-9]+]] = hl.declref( @r ): !hl.int
    // CHECK: [[V8:%[0-9]+]] = hl.implicit_cast( [[V7]] ): !hl.int
    // CHECK: return [[V8]] : !hl.int
    return r;
}

void add3()
{
    // CHECK: [[V1:%[0-9]+]] = hl.constant( 1 : i32 ): !hl.int
    // CHECK: [[V2:%[0-9]+]] = hl.constant( 2 : i32 ): !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.add %0, %1 : !hl.int
    // CHECK: [[V4:%[0-9]+]] = hl.var( v, %2 ): !hl.int
    int v = 1 + 2;
}
