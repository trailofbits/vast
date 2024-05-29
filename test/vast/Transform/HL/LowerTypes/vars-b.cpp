// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s

// CHECK-LABEL: hl.func @main {{.*}} () -> si32
int main()
{
    // CHECK: hl.var @x : !hl.lvalue<ui8> =  {
    // CHECK:   [[V1:%[0-9]+]] = hl.const #false
    // CHECK:   hl.value.yield [[V1]] : ui8
    // CHECK: }
    bool x = false;

    // CHECK: hl.var @y : !hl.lvalue<si8> =  {
    // CHECK:   [[V2:%[0-9]+]] = hl.const #core.integer<72> : si8
    // CHECK:   hl.value.yield [[V2]] : si8
    // CHECK: }
    char y = 'H';


    // CHECK: hl.var @ia : !hl.lvalue<!hl.array<3, si32>> =  {
    // CHECK:   [[V3:%[0-9]+]] = hl.const #core.integer<0> : si32
    // CHECK:   [[V4:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK:   [[V5:%[0-9]+]] = hl.const #core.integer<2> : si32
    // CHECK:   [[V6:%[0-9]+]] = hl.initlist [[V3]], [[V4]], [[V5]] : (si32, si32, si32) -> !hl.array<3, si32>
    // CHECK:   hl.value.yield [[V6]] : !hl.array<3, si32>
    // CHECK: }
    int ia[3] = { 0, 1, 2 };


    // CHECK: hl.var @ll : !hl.lvalue<si64> =  {
    // CHECK:   [[V7:%[0-9]+]] = hl.const #core.integer<142> : si32
    // CHECK:   [[V8:%[0-9]+]] = hl.implicit_cast [[V7]] IntegralCast : si32 -> si64
    // CHECK:   hl.value.yield [[V8]] : si64
    // CHECK: }
    long long ll = 142;

    // CHECK: hl.var @ld : !hl.lvalue<f128> =  {
    // CHECK:   [[V9:%[0-9]+]] = hl.const #core.float<91.019999999999996> : f64
    // CHECK:   [[V10:%[0-9]+]] = hl.implicit_cast [[V9]] FloatingCast : f64 -> f128
    // CHECK:   hl.value.yield [[V10]] : f128
    // CHECK: }
    long double ld = 91.02;

    // CHECK: hl.var @fa : !hl.lvalue<!hl.array<3, f32>> =  {
    // CHECK:   [[V11:%[0-9]+]] = hl.const #core.float<0.000000e+00> : f64
    // CHECK:   [[V12:%[0-9]+]] = hl.implicit_cast [[V11]] FloatingCast : f64 -> f32
    // CHECK:   [[V13:%[0-9]+]] = hl.const #core.float<5.000000e-01> : f64
    // CHECK:   [[V14:%[0-9]+]] = hl.implicit_cast [[V13]] FloatingCast : f64 -> f32
    // CHECK:   [[V15:%[0-9]+]] = hl.const #core.float<1.000000e+00> : f64
    // CHECK:   [[V16:%[0-9]+]] = hl.implicit_cast [[V15]] FloatingCast : f64 -> f32
    // CHECK:   [[V17:%[0-9]+]] = hl.initlist [[V12]], [[V14]], [[V16]] : (f32, f32, f32) -> !hl.array<3, f32>
    // CHECK:   hl.value.yield [[V17]] : !hl.array<3, f32>
    // CHECK: }
    float fa[3] = { 0.0, 0.5, 1.0 };
}
