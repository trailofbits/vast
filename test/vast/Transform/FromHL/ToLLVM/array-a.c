// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-to-ll-vars --vast-core-to-llvm | FileCheck %s

// REQUIRES: funcop-lowering

void count()
{
    // CHECK: llvm.store [[V4:%[0-9]+]], [[V10:%[0-9]+]] : !llvm.ptr<i32>
    int x[3] = { 112, 212, 4121 };
}
