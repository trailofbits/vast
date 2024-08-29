// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    int x = 0;
    typeof(x) y = 0;

    __auto_type u = x;
    __auto_type v = y;
    // CHECK: hl.cond : !hl.int
    // CHECK: hl.value.yield %{{[0-9]+}} : !hl.auto<!hl.int>
    // CHECK: hl.value.yield %{{[0-9]+}} : !hl.auto<!hl.typeof.expr<"(x)">>
    0 ? u : v;
}
