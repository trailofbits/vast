// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void minus(float arg1, float arg2) {

    // CHECK: [[R:%[0-9]+]] = hl.fsub [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.float, !hl.float) -> !hl.float
    float res = arg1 - arg2;
}
