// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-irs-to-llvm | %file-check %s

// CHECK: llvm.func @count([[ARG:%arg[0-9]+]]: i32)
void count(int arg)
{
    // CHECK: [[C:%[0-9]+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[V1:%[0-9]+]] = llvm.alloca [[C]] x i32 : (i64) -> !llvm.ptr<i32>
    // CHECK: [[V2:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.store [[V2]], [[V1]] : !llvm.ptr<i32>
    unsigned int iter = 0;
    // CHECK: [[V3:%[0-9]+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[V4:%[0-9]+]] = llvm.alloca [[V3]] x i32 : (i64) -> !llvm.ptr<i32>
    // CHECK: [[V5:%[0-9]+]] = llvm.mlir.constant(43 : i32) : i32
    // CHECK: llvm.store [[V5]], [[V1]] : !llvm.ptr<i32>
    // CHECK: llvm.store [[V5]], [[V4]] : !llvm.ptr<i32>
    unsigned int c = iter = 43;
    // CHECK: llvm.return
// CHECK:  }
}
