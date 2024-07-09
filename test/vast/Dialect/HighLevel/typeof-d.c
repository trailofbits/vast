// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

int main(void) {
    __typeof__(int) y = 0;
    // CHECK: hl.add {{.*}} (!hl.typeof.type<!hl.int>, !hl.int)
    __typeof__(0) x = y + 0;
    // CHECK: hl.add {{.*}} (!hl.int, !hl.typeof.expr<"(0)">)
    1 + x;
    // CHECK: hl.bin.ashr {{.*}} (!hl.typeof.expr<"(0)">, !hl.int)
    x >> 0;
    __typeof__(float) a = 0;
    __typeof__(a) b = 2;
    // CHECK: hl.fcmp {{.*}} !hl.typeof.type<!hl.float>, !hl.typeof.expr<"(a)">
    a == b;
    // CHECK: hl.fadd {{.*}} (!hl.typeof.expr<"(a)">, !hl.float)
    b + 3.0f;
}
