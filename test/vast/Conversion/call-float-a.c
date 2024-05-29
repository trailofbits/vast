// RUN: %check-emit-abi %s | %file-check %s -check-prefix=EMIT_ABI
// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// EMIT_ABI: abi.func @vast.abi.f_float{{.*}}([[A0:%.*]]: f32) -> none {
// EMIT_ABI: {{.*}} = abi.prologue {
// EMIT_ABI:   [[V2:%[0-9]+]] = abi.direct [[A0]] : f32 -> f32
// EMIT_ABI:   {{.*}}abi.yield [[V2]] : f32 -> f32
// EMIT_ABI: } : f32
void f_float(float a) {}

// EMIT_ABI: abi.func @vast.abi.f_double{{.*}}([[A0:%.*]]: f32) -> none {
// EMIT_ABI: {{.*}} = abi.prologue {
// EMIT_ABI:   [[V2:%[0-9]+]] = abi.direct [[A0]] : f32 -> f32
// EMIT_ABI:   {{.*}}abi.yield [[V2]] : f32 -> f32
// EMIT_ABI: } : f32
void f_double(float a) {}

// EMIT_ABI:  [[V1:%[0-9]+]] = hl.const #core.float<0.000000e+00> : f32
// EMIT_ABI:  [[V2:%[0-9]+]] = abi.call_exec @f_float([[V1]]) {
// EMIT_ABI:    [[V6:%[0-9]+]] = abi.call_args {
// EMIT_ABI:      {{.*}} = abi.direct [[V1]] : f32 -> f32
// EMIT_ABI:    } : f32
// EMIT_ABI:    [[V7:%[0-9]+]] = abi.call @f_float([[V6]]) : (f32) -> none
// EMIT_ABI:  } : (f32) -> none

// EMIT_ABI:  [[V4:%[0-9]+]] = hl.implicit_cast {{.*}} FloatingCast : f64 -> f32
// EMIT_ABI:  [[V5:%[0-9]+]] = abi.call_exec @f_double([[V4]]) {
// EMIT_ABI:    [[D6:%[0-9]+]] = abi.call_args {
// EMIT_ABI:      {{.*}} = abi.direct [[V4]] : f32 -> f32
// EMIT_ABI:    } : f32
// EMIT_ABI:    {{.*}} = abi.call @f_double([[D6]]) : (f32) -> none
// EMIT_ABI:  } : (f32) -> none


// VAL_CAT:  [[V1:%[0-9]+]] = hl.const #core.float<0.000000e+00> : f32
// VAL_CAT:  [[V2:%[0-9]+]] = hl.call @f_float([[V1]]) : (f32) -> none
// VAL_CAT:  [[V3:%[0-9]+]] = hl.const #core.float<5.000000e+00> : f64
// VAL_CAT:  [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] FloatingCast : f64 -> f32
// VAL_CAT:  [[V5:%[0-9]+]] = hl.call @f_double([[V4]]) : (f32) -> none


// C_LLVM: [[V0:%[0-9]+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// C_LLVM: llvm.call @f_float([[V0]]) : (f32) -> ()
// C_LLVM: [[V1:%[0-9]+]] = llvm.mlir.constant(5.000000e+00 : f64) : f64
// C_LLVM: [[V2:%[0-9]+]] = llvm.fptrunc [[V1]] : f64 to f32
// C_LLVM: llvm.call @f_double([[V2]]) : (f32) -> ()

int main()
{
    f_float(0.0f);
    f_double(5.0);
}
