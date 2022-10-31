// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-to-ll-vars --vast-core-to-llvm | FileCheck %s

// REQUIRES: funcop-lowering

void count()
{
    // CHECK: llvm.store [[V4:%[0-9]+]], [[V10:%[0-9]+]] : !llvm.ptr<f32>
    float x[3] = { 112.0f, 212.0f, 4121.0f };
}
