// RUN: %vast-cc1 -vast-emit-mlir-after=vast-lower-value-categories %s -o - | %file-check %s

enum E { A, B, C };

void shadowed_enum() {
    enum E { X, Y };
    // CHECK: ll.alloca
    // CHECK: hl.const #core.integer<1>
    enum E e = Y;
}
