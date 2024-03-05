// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-elaborated-types --vast-hl-lower-typedefs | %file-check %s
typedef int INT;

// CHECK: hl.func @fn ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int fn(INT x) {
    // CHECK: hl.ref [[A1]] : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
    return x;
}
