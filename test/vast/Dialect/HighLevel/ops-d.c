// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

void unary_inplace(int a) {
    // CHECK: hl.var "pre" : !hl.lvalue<!hl.int>
    // CHECK:  [[V1:%[0-9]+]] = hl.ref %arg0
    // CHECK:  [[V2:%[0-9]+]] = hl.pre.inc [[V1]] : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:  hl.value.yield [[V2]] : !hl.int
    int pre = ++a;

    // CHECK: hl.var "post" : !hl.lvalue<!hl.int>
    // CHECK:  [[V1:%[0-9]+]] = hl.ref %arg0
    // CHECK:  [[V2:%[0-9]+]] = hl.post.inc [[V1]] : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:  hl.value.yield [[V2]] : !hl.int
    int post = a++;
}
