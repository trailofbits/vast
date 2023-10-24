// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void plus(int arg1, int arg2) {

    // CHECK: [[R:%[0-9]+]] = hl.add [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.int, !hl.int) -> !hl.int
    int res = arg1 + arg2;
}
