// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @arithmetic
void arithmetic(int a)
{
    // CHECK: hl.post.inc
    a++;
    // CHECK: hl.pre.inc
    ++a;
    // CHECK: hl.post.dec
    a--;
    // CHECK: hl.pre.dec
    --a;
}

// CHECK-LABEL: func private @sign
void sign(int a)
{
    // CHECK: hl.plus
    +a;
    // CHECK: hl.minus
    -a;
}

// CHECK-LABEL: func private @binary
void binary(int a)
{
    // CHECK: hl.not
    ~a;
}

// CHECK-LABEL: func private @logical
void logical(bool a)
{
    // CHECK: hl.lnot
    !a;
}

