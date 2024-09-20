// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

struct foo {
    int bar;
};

void baz() {
    // compound literal: https://en.cppreference.com/w/c/language/compound_literal
    // CHECK: hl.compound_literal : !hl.lvalue<!hl.elaborated<!hl.record<@foo>>>
    (struct foo){};

    // these are actually lvalues
    // CHECK: hl.compound_literal : !hl.lvalue<!hl.elaborated<!hl.record<@foo>>>
    // CHECK: [[V1:%[0-9]+]] = hl.member {{.*}} at @bar : !hl.lvalue<!hl.elaborated<!hl.record<@foo>>> -> !hl.lvalue<!hl.int>
    // CHECK: hl.assign {{.*}} to [[V1]] : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
    ((struct foo){}).bar = 4;
    // CHECK: [[V2:%[0-9]+]] = hl.compound_literal : !hl.lvalue<!hl.elaborated<!hl.record<@foo>>>
    // CHECK: hl.addressof [[V2]]
    &(struct foo){};
}
