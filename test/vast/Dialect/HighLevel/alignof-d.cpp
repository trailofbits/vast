// RUN: %vast-cc1 -vast-emit-mlir=hl -Wno-gnu-statement-expression %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl -Wno-gnu-statement-expression %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.preferred_alignof.type !hl.int -> !hl.long< unsigned >
    unsigned long si = __alignof__(int);

    // CHECK: hl.var @v : !hl.lvalue<!hl.int>
    int v;

    // CHECK: hl.var @sv : !hl.lvalue<!hl.long< unsigned >>
    // CHECK: hl.preferred_alignof.expr -> !hl.long< unsigned >
    // CHECK: hl.ref @v
    unsigned long sv = __alignof__ v;
}
