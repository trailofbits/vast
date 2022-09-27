// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "a" : !hl.lvalue<!hl.int>
int a = 0;

// CHECK-LABEL: func @main
int main() {
    // CHECK: [[G:%[0-9]+]] = hl.globref "a" : !hl.lvalue<!hl.int>
    // CHECK: [[C:%[0-9]+]] = hl.const #hl.integer<1> : !hl.int
    // CHECK: hl.assign [[C]] to [[G]]
    a = 1;
}
