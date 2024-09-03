// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: [[L:%[0-9]+]] = hl.label.decl @end : !hl.label

    // CHECK: hl.var @x : !hl.lvalue<!hl.int>
    int x;

    // CHECK: hl.goto [[L]]
    goto end;

    // CHECK: hl.label [[L]]
    end:;
}
