// RUN: vast-cc --from-source %s | FileCheck %s

void arithemtic_signed(int a, int b)
{
    int c;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast( [[V1]] )
    // CHECK: [[V3:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast( [[V3]] )
    // CHECK: hl.add [[V2]], [[V4]]
    c = a + b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast( [[V1]] )
    // CHECK: [[V3:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast( [[V3]] )
    // CHECK: hl.sub [[V2]], [[V4]]
    c = a - b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast( [[V1]] )
    // CHECK: [[V3:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast( [[V3]] )
    // CHECK: hl.mul [[V2]], [[V4]]
    c = a * b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast( [[V1]] )
    // CHECK: [[V3:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast( [[V3]] )
    // CHECK: hl.sdiv [[V2]], [[V4]]
    c = a / b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast( [[V1]] )
    // CHECK: [[V3:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast( [[V3]] )
    // CHECK: hl.srem [[V2]], [[V4]]
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

void assign_signed(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast( [[V2]] )
    // CHECK: hl.assign.add [[V3]] → [[V1]]
    a += b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast( [[V2]] )
    // CHECK: hl.assign.sub [[V3]] → [[V1]]
    a -= b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast( [[V2]] )
    // CHECK: hl.assign.mul [[V3]] → [[V1]]
    a *= b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast( [[V2]] )
    // CHECK: hl.assign.sdiv [[V3]] → [[V1]]
    a /= b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast( [[V2]] )
    // CHECK: hl.assign.srem [[V3]] → [[V1]]
    a %= b;
}

void assign_unsigned(unsigned a, unsigned b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast( [[V2]] )
    // CHECK: hl.assign.udiv [[V3]] → [[V1]]
    a /= b;
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @a )
    // CHECK: [[V2:%[0-9]+]] = hl.declref( @b )
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast( [[V2]] )
    // CHECK: hl.assign.urem [[V3]] → [[V1]]
    a %= b;
}
