// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z10arithmetici
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

// CHECK-LABEL: hl.func @_Z4signi
void sign(int a)
{
    // CHECK: hl.plus
    +a;
    // CHECK: hl.minus
    -a;
}

// CHECK-LABEL: hl.func @_Z6binaryi
void binary(int a)
{
    // CHECK: hl.not
    ~a;
}

// CHECK-LABEL: hl.func @_Z7logicalb
void logical(bool a)
{
    // CHECK: hl.lnot
    !a;
}

