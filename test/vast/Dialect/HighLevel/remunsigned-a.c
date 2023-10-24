// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void div(unsigned int arg1, unsigned int arg2) {

    // CHECK: [[R:%[0-9]+]] = hl.urem [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.int< unsigned >, !hl.int< unsigned >) -> !hl.int< unsigned >
    unsigned int res = arg1 % arg2;
}
