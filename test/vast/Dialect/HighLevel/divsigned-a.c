// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

void div(int arg1, int arg2) {

    // CHECK: [[R:%[0-9]+]] = hl.sdiv [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.int, !hl.int) -> !hl.int
    int res = arg1 / arg2;
}
