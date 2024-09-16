// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o -  | %file-check %s

void count()
{
    // CHECK: llvm.store {{.*}} : !llvm.array<3 x f32>, !llvm.ptr
    float x[3] = { 112.0f, 212.0f, 4121.0f };
}
