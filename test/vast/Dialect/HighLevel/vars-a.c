// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: func @main() -> !hl.int
int main()
{
    // CHECK: hl.var @a : !hl.int
    int a;

    // CHECK: hl.var @b : !hl.int = {
    // CHECK:   [[V1:%[0-9]+]] = hl.constant(#hl.int<1>)
    // CHECK:   hl.value.yield [[V1]]
    // CHECK: }
    int b = 1;

    unsigned int ub = 1U;

    // CHECK: hl.var @c : !hl.int = {
    // CHECK:   [[V2:%[0-9]+]] = hl.constant(#hl.int<1>)
    // CHECK:   hl.value.yield [[V2]]
    // CHECK: }
    int c( 1 );

    int ni = -1;

    long nl = -1;
}
