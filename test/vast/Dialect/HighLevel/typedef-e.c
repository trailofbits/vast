// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

typedef float V __attribute__((vector_size (4 * sizeof (float))));

void fun() {
    V x;
    // CHECK: hl.subscript {{.*}} !hl.lvalue<!hl.elaborated<!hl.typedef<"V">>>
    float a = x[0];
}

