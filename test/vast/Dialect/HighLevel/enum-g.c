// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.enum @E
// CHECK:   hl.enum.const @A
// CHECK:   hl.enum.const @B
// CHECK:   hl.enum.const @C
enum E { A, B, C };

void shadowed_enum() {
    // CHECK: hl.enum @E
    // CHECK:   hl.enum.const @X
    enum E { X };
    // CHECK: hl.var @e
    // CHECK:   hl.enumref @X
    enum E e = X;
}
