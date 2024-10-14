// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.struct @bitfield
struct bitfield {
    // CHECK: hl.field @x bw 3 : !hl.int< unsigned >
    unsigned x: 3;
};

void foo() {
    // CHECK: hl.var @a : !hl.lvalue<!hl.array<2, !hl.int>>
    int a[2];
    // CHECK: hl.var @i : !hl.lvalue<!hl.int>
    int i;
    // CHECK: hl.var @j constant : !hl.lvalue<!hl.int< const >>
    const int j;
    // CHECK: hl.var @bf : !hl.lvalue<!hl.elaborated<!hl.record<@bitfield>>>
    struct bitfield bf;

    // these are all lvalues
    // CHECK: hl.ref @a
    a;
    // CHECK: hl.ref @i
    i;
    // CHECK: hl.ref @j
    j;
    // CHECK: [[R:%[0-9]+]] = hl.ref @bf
    // CHECK: hl.member [[R]] at @x : !hl.lvalue<!hl.elaborated<!hl.record<@bitfield>>> -> !hl.lvalue<!hl.int< unsigned >>
    bf.x;

    // CHECK: [[F:%[0-9]+]] = hl.funcref @foo : !core.fn<() -> (!hl.void)>
    foo;

    // CHECK: hl.implicit_cast [[F]] FunctionToPointerDecay
    &foo;
}
