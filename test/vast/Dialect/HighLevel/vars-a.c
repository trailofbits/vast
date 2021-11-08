// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func @main() -> !hl.int
int main()
{
    // CHECK: hl.var @a : !hl.int
    int a;

    // CHECK: [[V1:%[0-9]+]] = hl.constant 1
    // CHECK: hl.var @b = [[V1]] : !hl.int
    int b = 1;

    // CHECK: [[V2:%[0-9]+]] = hl.constant 1
    // CHECK: hl.var @c = [[V2]] : !hl.int
    int c( 1 );
}
