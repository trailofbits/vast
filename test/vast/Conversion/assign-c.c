// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

// CHECK: llvm.func @count
void count()
{
    // CHECK: [[V0:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[V1:%[0-9]+]] = llvm.alloca [[V0]] x i32 : (i64) -> !llvm.ptr
    // CHECK: [[V2:%[0-9]+]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK: llvm.store [[V2]], [[V1]] : i32, !llvm.ptr
    // CHECK: [[V3:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[V4:%[0-9]+]] = llvm.alloca [[V3]] x i32 : (i64) -> !llvm.ptr
    // CHECK: [[V5:%[0-9]+]] = llvm.load [[V1]] : !llvm.ptr -> i32
    // CHECK: llvm.store [[V5]], [[V4]] : i32, !llvm.ptr

    unsigned int c = 15, iter = c;
    // CHECK: llvm.return
}
