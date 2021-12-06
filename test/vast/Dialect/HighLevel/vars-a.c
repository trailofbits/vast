// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: func @main() -> !hl.int
int main()
{
    // CHECK: hl.var @a : !hl.int
    int a;

    // CHECK: hl.var @b : !hl.int = {
    // CHECK:   [[V1:%[0-9]+]] = hl.constant 1
    // CHECK:   hl.value.yield [[V1]]
    // CHECK: }
    int b = 1;

    // CHECK: hl.var @c : !hl.int = {
    // CHECK:   [[V2:%[0-9]+]] = hl.constant 1
    // CHECK:   hl.value.yield [[V2]]
    // CHECK: }
    int c( 1 );
}
