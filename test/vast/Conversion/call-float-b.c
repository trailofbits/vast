// RUN: %check-emit-abi %s | %file-check %s -check-prefix=EMIT_ABI
// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// EMIT_ABI:  abi.func @vast.abi.f_float{{.*}}() -> f32 {
// EMIT_ABI:    [[V0:%[0-9]+]] = hl.const #core.float<1.000000e-01> : f32
// EMIT_ABI:    ll.return {{.*}} : f32
// EMIT_ABI:  }
float f_float() { return 0.1f; }

// EMIT_ABI:  abi.func @vast.abi.f_double{{.*}}() -> f64 {
// EMIT_ABI:    [[V0:%[0-9]+]] = hl.const #core.float<5.000000e+00> : f64
// EMIT_ABI:    ll.return {{.*}} : f64
// EMIT_ABI:  }
double f_double() { return 5.0; }

// EMIT_ABI:  {{.*}} = abi.call_exec @f_float() {
// EMIT_ABI:    [[F3:%[0-9]+]] = abi.call @f_float() : () -> f32
// EMIT_ABI:    [[F4:%[0-9]+]] = abi.call_rets {
// EMIT_ABI:      {{.*}} = abi.direct [[F3]] : f32 -> f32
// EMIT_ABI:    } : f32
// EMIT_ABI:    {{.*}} = abi.yield [[F4]] : f32 -> f32
// EMIT_ABI:  } : () -> f32

// EMIT_ABI:  {{.*}} = abi.call_exec @f_double() {
// EMIT_ABI:    [[D3:%[0-9]+]] = abi.call @f_double() : () -> f64
// EMIT_ABI:    [[D4:%[0-9]+]] = abi.call_rets {
// EMIT_ABI:      {{.*}} = abi.direct [[D3]] : f64 -> f64
// EMIT_ABI:    } : f64
// EMIT_ABI:    {{.*}} = abi.yield [[D4]] : f64 -> f64
// EMIT_ABI:  } : () -> f64


// VAL_CAT:  {{.*}} = hl.call @f_float() : () -> f32
// VAL_CAT:  {{.*}} = hl.call @f_double() : () -> f64


// C_LLVM: {{.*}} = llvm.call @f_float() : () -> f32
// C_LLVM: {{.*}} = llvm.call @f_double() : () -> f64

int main()
{
    f_float();
    f_double();
}
