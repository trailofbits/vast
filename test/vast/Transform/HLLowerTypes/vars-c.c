// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types | FileCheck %s

// CHECK-LABEL: func @main() -> i32
int main()
{
    // CHECK: hl.var "x" : i32 =  {
    // CHECK:   [[V1:%[0-9]+]] = hl.constant.int 0 : i32
    // CHECK:   hl.value.yield [[V1]] : i32
    int x = 0;
    // CHECK: hl.var "cx" : i32 =  {
    // CHECK:   [[V2:%[0-9]+]] = hl.constant.int 0 : i32
    // CHECK:   hl.value.yield [[V2]] : i32
    const int cx = 0;
    // CHECK: hl.var "cvx" : i32 =  {
    // CHECK:   [[V3:%[0-9]+]] = hl.constant.int 0 : i32
    // CHECK:   hl.value.yield [[V3]] : i32
    const volatile int cvx = 0;
}
