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

void fun2(FLOAT a, float b) {
    // CHECK: hl.fadd {{.*}}!hl.elaborated<!hl.typedef<"FLOAT">>, !hl.float
    float c = a + b;
    // CHECK: hl.fcmp olt {{.*}}!hl.elaborated<!hl.typedef<"FLOAT">>, !hl.float
    if (a < b)
        return;
    return;
}

