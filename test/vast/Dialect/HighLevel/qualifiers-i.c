// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.union @u
union u { int i; double d; };

// CHECK: hl.var @v : !hl.lvalue<!hl.elaborated<!hl.record<"u">>>
union u v;

// CHECK: hl.var @cv : !hl.lvalue<!hl.elaborated<!hl.record<"u">,  const >>
const union u cv;

// CHECK: hl.var @cvv : !hl.lvalue<!hl.elaborated<!hl.record<"u">,  const, volatile >>
const volatile union u cvv;

// CHECK: hl.typedef @e : !hl.elaborated<!hl.record<"u">>
typedef union u e;

// CHECK: hl.var @v : !hl.lvalue<!hl.elaborated<!hl.typedef<"e">>>
e v;

// CHECK: hl.var @cv : !hl.lvalue<!hl.elaborated<!hl.typedef<"e">,  const >>
const e cv;

// CHECK: hl.var @vv : !hl.lvalue<!hl.elaborated<!hl.typedef<"e">,  volatile >>
volatile e vv;

// CHECK: hl.var @cvv : !hl.lvalue<!hl.elaborated<!hl.typedef<"e">,  const, volatile >>
const volatile e cvv;
