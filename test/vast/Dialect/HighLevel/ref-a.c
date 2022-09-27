// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    // CHECK: [[X:%[0-9]+]] = hl.var "x" : !hl.lvalue<!hl.int> = {
    // CHECK: [[V1:%[0-9]+]] = hl.const #hl.integer<0> : !hl.int
    int x = 0;
    // CHECK: [[Y:%[0-9]+]] = hl.var "y" : !hl.lvalue<!hl.int> = {
    // CHECK: [[V2:%[0-9]+]] = hl.ref [[X]] : !hl.lvalue<!hl.int>
    int y = x;
}
