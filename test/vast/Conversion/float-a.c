// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

void count()
{
    // CHECK: [[V0:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[V1:%[0-9]+]] = llvm.alloca [[V0]] x f32 : (i64) -> !llvm.ptr
    // CHECK: [[V2:%[0-9]+]] = llvm.mlir.constant(2.000000e-01 : f32) : f32
    // CHECK: llvm.store [[V2]], [[V1]] : f32, !llvm.ptr
    float fa = 0.2f;

    // CHECK: [[V3:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[V4:%[0-9]+]] = llvm.alloca [[V3]] x f64 : (i64) -> !llvm.ptr
    // CHECK: [[V5:%[0-9]+]] = llvm.mlir.constant(5.512000e+01 : f64) : f64
    // CHECK: llvm.store [[V5]], [[V4]] : f64, !llvm.ptr
    double da = 55.12;
}
