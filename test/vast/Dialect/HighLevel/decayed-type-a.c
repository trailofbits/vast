// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @f {{.*}} ({{.*}}: !hl.lvalue<!hl.decayed<!hl.ptr<!hl.int>>>)
void f(int i[]) {
    // CHECK: hl.subscript {{.*}} at [{{.*}} : !hl.int] : !hl.decayed<!hl.ptr<!hl.int>> -> !hl.lvalue<!hl.int>
    i[1] = 0;
}
