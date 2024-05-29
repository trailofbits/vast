// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.stmt.expr : !hl.int
    // CHECK:   hl.var @x
    // CHECK:   hl.value.yield
    // CHECK:   hl.value.yield
    int v = ({int x = 4; x;});
}
