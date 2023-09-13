// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @f ({{.*}}: !hl.lvalue<!hl.decayed<!hl.ptr<!hl.int>>>)
void f(int i[]) {
    // CHECK: hl.subscript {{.*}} at [{{.*}} : !hl.int] : !hl.decayed<!hl.ptr<!hl.int>> -> !hl.lvalue<!hl.int>
    i[1] = 0;
}
