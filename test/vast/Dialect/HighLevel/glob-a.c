// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @a : !hl.lvalue<!hl.int>
int a = 0;

// CHECK-LABEL: hl.func @main
int main() {
    // CHECK: [[G:%[0-9]+]] = hl.globref "a" : !hl.lvalue<!hl.int>
    // CHECK: [[C:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK: hl.assign [[C]] to [[G]]
    a = 1;
}
