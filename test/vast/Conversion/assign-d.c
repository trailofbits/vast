// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

void count()
{
    // CHECK: [[V0:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[V1:%[0-9]+]] = llvm.alloca [[V0]] x i32 : (i64) -> !llvm.ptr
    // CHECK: [[V2:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: llvm.store [[V2]], [[V1]] : i32, !llvm.ptr
    int x = 1;
    // CHECK: [[V3:%[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: [[V4:%[0-9]+]] = llvm.load [[V1]] : !llvm.ptr -> i32
    // CHECK: [[V5:%[0-9]+]] = llvm.sub [[V4]], [[V3]]  : i32
    // CHECK: llvm.store [[V5]], [[V1]] : i32, !llvm.ptr
    x -= 2;
    // CHECK: [[V6:%[0-9]+]] = llvm.load [[V1]] : !llvm.ptr -> i32
    // CHECK: [[V7:%[0-9]+]] = llvm.load [[V1]] : !llvm.ptr -> i32
    // CHECK: [[V8:%[0-9]+]] = llvm.sub [[V7]], [[V6]]  : i32
    // CHECK: llvm.store [[V8]], [[V1]] : i32, !llvm.ptr
    x -= x;

    // CHECK: llvm.return
}
