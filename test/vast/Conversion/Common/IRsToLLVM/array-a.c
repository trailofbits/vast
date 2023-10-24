// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-to-ll-cf --vast-hl-to-ll-vars --vast-irs-to-llvm | %file-check %s

void count()
{
    // CHECK: llvm.store [[V4:%[0-9]+]], [[V10:%[0-9]+]] : !llvm.ptr<array<3 x i32>>
    int x[3] = { 112, 212, 4121 };
}
