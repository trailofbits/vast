// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "a" : !hl.lvalue<!hl.int>
int a = 0;

// CHECK-LABEL: func @main
int main() {
    // CHECK: [[G:%[0-9]+]] = hl.global.ref "a" : !hl.lvalue<!hl.int>
    // CHECK: hl.decl.ref [[G]] : !hl.lvalue<!hl.int>
    a = 1;
}
