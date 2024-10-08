// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.sizeof.type !hl.int -> !hl.long< unsigned >
    unsigned long si = sizeof(int);

    int v;

    // CHECK: hl.var @v : !hl.lvalue<!hl.int>
    // CHECK: hl.var @sv : !hl.lvalue<!hl.long< unsigned >>
    // CHECK: hl.sizeof.expr -> !hl.long< unsigned >
    // CHECK:  hl.ref @v
    unsigned long sv = sizeof v;
}
