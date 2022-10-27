// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-to-ll-vars --vast-core-to-llvm | FileCheck %s

// REQUIRES: funcop-lowering

void count()
{
    // CHECK: [[V0:%[0-9]+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[V1:%[0-9]+]] = llvm.alloca [[V0]] x f32 : (i64) -> !llvm.ptr<f32>
    // CHECK: [[V2:%[0-9]+]] = llvm.mlir.constant(2.000000e-01 : f32) : f32
    // CHECK: llvm.store [[V2]], [[V1]] : !llvm.ptr<f32>
    float fa = 0.2f;

    // CHECK: [[V3:%[0-9]+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[V4:%[0-9]+]] = llvm.alloca [[V3]] x f64 : (i64) -> !llvm.ptr<f64>
    // CHECK: [[V5:%[0-9]+]] = llvm.mlir.constant(5.512000e+01 : f64) : f64
    // CHECK: llvm.store [[V5]], [[V4]] : !llvm.ptr<f64>
    double da = 55.12;
}
