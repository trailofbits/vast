// RUN: %vast-cc1 -vast-emit-mlir=llvm %s -o - | %file-check %s

// CHECK: llvm.func @fn(%arg0: f32, %arg1: f32) -> f32 {
float fn(float arg0, float arg1)
{
    // CHECK: [[R:%[0-9]+]] = llvm.fdiv [[V1:%[0-9]+]], [[V2:%[0-9]+]] : f32
    // CHECK: llvm.return [[R]] : f32
    return arg0 / arg1;
}
// CHECK : }
