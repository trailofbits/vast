// RUN: %vast-cc1 -vast-emit-mlir=hl -Wno-gnu-statement-expression %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl -Wno-gnu-statement-expression %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.alignof.type !hl.int -> !hl.long< unsigned >
    unsigned long si = alignof(int);

    // CHECK: [[V1:%[0-9]+]] = hl.var @v : !hl.lvalue<!hl.int>
    int v;

    // CHECK: hl.var @sv : !hl.lvalue<!hl.long< unsigned >>
    // CHECK: hl.alignof.expr -> !hl.long< unsigned >
    // CHECK: hl.ref [[V1]]
    unsigned long sv = alignof v;
}
