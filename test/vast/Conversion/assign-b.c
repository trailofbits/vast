// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

// CHECK: llvm.func @count
void count()
{
    // CHECK: [[C:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[V1:%[0-9]+]] = llvm.alloca [[C]] x i32 : (i64) -> !llvm.ptr
    // CHECK: [[V2:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.store [[V2]], [[V1]] : i32, !llvm.ptr
    unsigned int iter = 0;
    // CHECK: [[V3:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[V4:%[0-9]+]] = llvm.alloca [[V3]] x i32 : (i64) -> !llvm.ptr
    // CHECK: [[V5:%[0-9]+]] = llvm.mlir.constant(43 : i32) : i32
    // CHECK: llvm.store [[V5]], [[V1]] : i32, !llvm.ptr
    // CHECK: llvm.store [[V5]], [[V4]] : i32, !llvm.ptr
    unsigned int c = iter = 43;
    // CHECK: llvm.return
// CHECK:  }
}
