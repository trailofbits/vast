// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt --vast-hl-lower-types %t | diff -B %t -

// CHECK-LABEL: func @main() -> i32
int main()
{
    // CHECK: hl.var @a : i32
    int a;

    // CHECK: hl.var @b : i32 = {
    // CHECK:   [[V1:%[0-9]+]] = hl.constant.int 1 : i32
    // CHECK:   hl.value.yield [[V1]] : i32
    // CHECK: }
    int b = 1;

    unsigned int ub = 1U;

    // CHECK: hl.var @c : i32 = {
    // CHECK:   [[V2:%[0-9]+]] = hl.constant.int 1 : i32
    // CHECK:   hl.value.yield [[V2]] : i32
    // CHECK: }
    int c( 1 );

    int ni = -1;

    // CHECK: hl.implicit_cast [[V3:%[0-9]+]] IntegralCast: i32 -> i64
    long nl = -1;
}
