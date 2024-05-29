// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main()
{
    // CHECK: [[X:%[0-9]+]] = hl.var @x : !hl.lvalue<!hl.int> = {
    // CHECK: [[V1:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
    int x = 0;
    // CHECK: [[Y:%[0-9]+]] = hl.var @y : !hl.lvalue<!hl.int> = {
    // CHECK: [[V2:%[0-9]+]] = hl.ref [[X]]
    int y = x;
}
