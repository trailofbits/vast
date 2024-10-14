// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void scope() {
    // CHECK: hl.var @i : !hl.lvalue<!hl.int>
    int i;

    // CHECK: hl.var @u : !hl.lvalue<!hl.int< unsigned >>
    unsigned u;

    // CHECK: hl.var @s : !hl.lvalue<!hl.int>
    signed s;

    // CHECK: hl.var @ui : !hl.lvalue<!hl.int< unsigned >>
    unsigned int ui;

    // CHECK: hl.var @us : !hl.lvalue<!hl.short< unsigned >>
    unsigned short us;

    // CHECK: hl.var @ci constant : !hl.lvalue<!hl.int< const >> = {
    // CHECK: [[C1:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
    // CHECK: hl.value.yield [[C1]]
    const int ci = 0;

    // CHECK: hl.var @cui constant : !hl.lvalue<!hl.int< unsigned, const >> = {
    // CHECK: [[C2:%[0-9]+]] = hl.const #core.integer<0> : !hl.int< unsigned >
    // CHECK: hl.value.yield [[C2]]
    const unsigned cui = 0U;

    // CHECK: hl.var @vi : !hl.lvalue<!hl.int< volatile >>
    volatile int vi;

    // CHECK: hl.var @vui : !hl.lvalue<!hl.int< unsigned, volatile >>
    volatile unsigned vui;

    // CHECK: hl.var @cvi constant : !hl.lvalue<!hl.int< const, volatile >> = {
    // CHECK: [[C3:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
    // CHECK: hl.value.yield [[C3]]
    const volatile int cvi = 0;

    // CHECK: hl.var @cvui constant : !hl.lvalue<!hl.int< unsigned, const, volatile >> = {
    // CHECK: [[C4:%[0-9]+]] = hl.const #core.integer<0> : !hl.int< unsigned >
    // CHECK: hl.value.yield [[C4]]
    const volatile unsigned int cvui = 0U;

    // CHECK: hl.var @b : !hl.lvalue<!hl.bool>
    bool b;

    // CHECK: hl.var @vb : !hl.lvalue<!hl.bool< volatile >>
    volatile bool vb;

    // CHECK: hl.var @cb constant : !hl.lvalue<!hl.bool< const >> = {
    // CHECK: [[C5:%[0-9]+]] = hl.const #false
    // CHECK: hl.value.yield [[C5]]
    const bool cb = false;

    // CHECK: hl.var @cvb constant : !hl.lvalue<!hl.bool< const, volatile >> = {
    // CHECK: [[C6:%[0-9]+]] = hl.const #true
    // CHECK: hl.value.yield [[C6]]
    const volatile bool cvb = true;
}
