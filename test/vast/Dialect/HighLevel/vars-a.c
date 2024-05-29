// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main()
{
    // CHECK: hl.var @a : !hl.lvalue<!hl.int>
    int a;

    // CHECK: hl.var @b : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V1:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK:   hl.value.yield [[V1]]
    // CHECK: }
    int b = 1;

    // CHECK: hl.var @ub : !hl.lvalue<!hl.int< unsigned >> = {
    // CHECK:   [[V2:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK:   hl.value.yield [[V2]]
    // CHECK: }
    unsigned int ub = 1U;

    // CHECK: hl.var @ni : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V3:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK:   [[V4:%[0-9]+]] = hl.minus [[V3]] : !hl.int
    // CHECK:   hl.value.yield [[V4]]
    // CHECK: }
    int ni = -1;


    // CHECK: hl.var @nl : !hl.lvalue<!hl.long> = {
    // CHECK:   [[V5:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK:   [[V6:%[0-9]+]] = hl.minus [[V5]] : !hl.int
    // CHECK:   [[V7:%[0-9]+]] = hl.implicit_cast [[V6]] IntegralCast : !hl.int -> !hl.long
    // CHECK:   hl.value.yield [[V7]]
    // CHECK: }
    long nl = -1;
}
