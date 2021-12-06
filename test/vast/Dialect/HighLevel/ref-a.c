// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    // CHECK: hl.var @x : !hl.int = {
    // CHECK: [[V1:%[0-9]+]] = hl.constant 0 : !hl.int
    int x = 0;
    // CHECK: hl.var @y : !hl.int = {
    // CHECK: [[V2:%[0-9]+]] = hl.declref @x : !hl.int
    int y = x;
}
