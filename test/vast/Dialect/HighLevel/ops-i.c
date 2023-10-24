// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o %t %s && %vast-opt %t | diff -B %t -

int main() {
    int a, b;
    unsigned ua, ub;
    float fa, fb;

    // CHECK: hl.assign.add
    a += b;
    // CHECK: hl.assign.sub
    a -= b;
    // CHECK: hl.assign.mul
    a *= b;
    // CHECK: hl.assign.sdiv
    a /= b;
    // CHECK: hl.assign.srem
    a %= b;
    // CHECK: hl.assign.bin.or
    a |= b;
    // CHECK: hl.assign.bin.and
    a &= b;
    // CHECK: hl.assign.bin.xor
    a ^= b;
    // CHECK: hl.assign.bin.shl
    a <<= b;
    // CHECK: hl.assign.bin.ashr
    a >>= b;

    // CHECK: hl.assign.add
    ua += ub;
    // CHECK: hl.assign.sub
    ua -= ub;
    // CHECK: hl.assign.mul
    ua *= ub;
    // CHECK: hl.assign.udiv
    ua /= ub;
    // CHECK: hl.assign.urem
    ua %= ub;
    // CHECK: hl.assign.bin.lshr
    ua >>= ub;

    // CHECK: hl.assign.fadd
    fa += fb;
    // CHECK: hl.assign.fsub
    fa -= fb;
    // CHECK: hl.assign.fmul
    fa *= fb;
    // CHECK: hl.assign.fdiv
    fa /= fb;
}
