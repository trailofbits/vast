// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// C_LLVM: llvm.func @fn([[A0:%.*]]: f32, [[A1:%.*]]: f32) -> f32 {
// C_LLVM:  [[R:%[0-9]+]] = llvm.fadd [[V1:%[0-9]+]], [[V2:%[0-9]+]] : f32
// C_LLVM:  llvm.return [[R]] : f32
// C_LLVM: }
float fn(float arg0, float arg1)
{
    return arg0 + arg1;
}
