// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

void plus(float arg1, float arg2) {

    // CHECK: [[R:%[0-9]+]] = hl.fsub [[V1:%[0-9]+]], [[V2:%[0-9]+]] : !hl.float
    float res = arg1 - arg2;
}
