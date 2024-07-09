// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

int main(void) {
    int a;
    __typeof__(int) y = 0;
    // CHECK: hl.add {{.*}} (!hl.typeof.type<!hl.int>, !hl.int)
    __typeof__(0) x = y + 0;
    // CHECK: hl.add {{.*}} (!hl.int, !hl.typeof.expr<"(0)">)
    return 1 + x;
}
