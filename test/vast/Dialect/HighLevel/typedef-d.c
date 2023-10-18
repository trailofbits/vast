// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.typedef "FLOAT" : !hl.float
typedef float FLOAT;
void fun(FLOAT a, FLOAT b) {
    // CHECK: hl.fcmp olt
    if (a < b)
        return;
    return;
}

