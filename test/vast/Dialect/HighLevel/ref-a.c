// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    // CHECK: [[V1:%[0-9]+]] = hl.constant 0 : !hl.int
    // CHECK: hl.var @x = [[V1]] : !hl.int
    int x = 0;
    // CHECK: [[V2:%[0-9]+]] = hl.declref @x : !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.int -> !hl.int
    // CHECK: hl.var @y = [[V3]] : !hl.int
    int y = x;
}
