// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -m16 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -m32 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -mx32 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void minus(int arg1, int arg2) {

    // CHECK: [[R:%[0-9]+]] = hl.sub [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.int, !hl.int) -> !hl.int
    int res = arg1 - arg2;
}
