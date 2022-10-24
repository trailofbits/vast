// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func external @arithmetic
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

// CHECK-LABEL: hl.func external @sign
void sign(int a)
{
    // CHECK: hl.plus
    +a;
    // CHECK: hl.minus
    -a;
}

// CHECK-LABEL: hl.func external @binary
void binary(int a)
{
    // CHECK: hl.not
    ~a;
}

// CHECK-LABEL: hl.func external @logical
void logical(bool a)
{
    // CHECK: hl.lnot
    !a;
}

