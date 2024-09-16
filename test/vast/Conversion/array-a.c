// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

void count()
{
    // CHECK: llvm.store {{.*}} : !llvm.array<3 x i32>, !llvm.ptr
    int x[3] = { 112, 212, 4121 };
}
