// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.struct @foo
struct foo {
    // CHECK: hl.field @x : !hl.int
    // CHECK: hl.field @y : !hl.int
    int x, y;
};

// CHECK: hl.struct @lots_of_inits
struct lots_of_inits {
    // CHECK: hl.field @z : !hl.array<2, !hl.elaborated<!hl.record<@foo>>>
    struct foo z[2];
    // CHECK: hl.field @w : !hl.array<3, !hl.int>
    int w[3];
};

// CHECK: hl.var @init : !hl.lvalue<!hl.elaborated<!hl.record<@lots_of_inits>>>
struct lots_of_inits init = {
    // CHECK: [[A:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK: [[B:%[0-9]+]] = hl.const #core.integer<2> : !hl.int
    // CHECK: [[F1:%[0-9]+]] = hl.initlist [[A]], [[B]] : (!hl.int, !hl.int) -> !hl.elaborated<!hl.record<@foo>>

    // CHECK: [[C:%[0-9]+]] = hl.const #core.integer<3> : !hl.int
    // CHECK: [[D:%[0-9]+]] = hl.const #core.integer<4> : !hl.int
    // CHECK: [[F2:%[0-9]+]] = hl.initlist [[C]], [[D]] : (!hl.int, !hl.int) -> !hl.elaborated<!hl.record<@foo>>

    // CHECK: hl.initlist [[IA:%[0-9]+]], [[IB:%[0-9]+]] : (!hl.elaborated<!hl.record<@foo>>, !hl.elaborated<!hl.record<@foo>>) -> !hl.array<2, !hl.elaborated<!hl.record<@foo>>>

    // CHECK: hl.initlist [[X:%[0-9]+]], [[Y:%[0-9]+]], [[Z:%[0-9]+]] : (!hl.int, !hl.int, !hl.int) -> !hl.array<3, !hl.int>
    {{1, 2}, {3, 4}}, {5, 6, 7}
};

// CHECK: hl.var @flat_init : !hl.lvalue<!hl.elaborated<!hl.record<@lots_of_inits>>>
struct lots_of_inits flat_init = {

    // CHECK: [[A:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK: [[B:%[0-9]+]] = hl.const #core.integer<2> : !hl.int
    // CHECK: [[F1:%[0-9]+]] = hl.initlist [[A]], [[B]] : (!hl.int, !hl.int) -> !hl.elaborated<!hl.record<@foo>>

    // CHECK: [[C:%[0-9]+]] = hl.const #core.integer<3> : !hl.int
    // CHECK: [[D:%[0-9]+]] = hl.const #core.integer<4> : !hl.int
    // CHECK: [[F2:%[0-9]+]] = hl.initlist [[C]], [[D]] : (!hl.int, !hl.int) -> !hl.elaborated<!hl.record<@foo>>

    // CHECK: hl.initlist [[IA:%[0-9]+]], [[IB:%[0-9]+]] : (!hl.elaborated<!hl.record<@foo>>, !hl.elaborated<!hl.record<@foo>>) -> !hl.array<2, !hl.elaborated<!hl.record<@foo>>>

    // CHECK: hl.initlist [[X:%[0-9]+]], [[Y:%[0-9]+]], [[Z:%[0-9]+]] : (!hl.int, !hl.int, !hl.int) -> !hl.array<3, !hl.int>
    1, 2, 3, 4, 5, 6, 7
};
