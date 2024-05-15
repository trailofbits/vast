// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-irs-to-llvm | %file-check %s

// CHECK: llvm.func @fn(%arg0: i32, %arg1: i32) -> i32 {
int fn(int arg0, int arg1)
{
    // CHECK: [[R:%[0-9]+]] = llvm.sdiv [[V1:%[0-9]+]], [[V2:%[0-9]+]] : i32
    // CHECK: llvm.return [[R]] : i32
    return arg0 / arg1;
}
// CHECK : }
