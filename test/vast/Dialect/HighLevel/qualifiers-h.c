// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.enum @e : !hl.int< unsigned >
enum e { a, b, c };

// CHECK: hl.var @v, <external> : !hl.lvalue<!hl.elaborated<!hl.enum<@e>>>
enum e v;

// CHECK: hl.var @cv, <external> constant : !hl.lvalue<!hl.elaborated<!hl.enum<@e>,  const >>
const enum e cv;

// CHECK: hl.var @cvv, <external> constant : !hl.lvalue<!hl.elaborated<!hl.enum<@e>,  const, volatile >>
const volatile enum e cvv;

// CHECK: hl.typedef @def : !hl.elaborated<!hl.enum<@e>>
typedef enum e def;

// CHECK: hl.var @d, <external> : !hl.lvalue<!hl.elaborated<!hl.typedef<@def>>>
def d;

// CHECK: hl.var @cd, <external> constant : !hl.lvalue<!hl.elaborated<!hl.typedef<@def>,  const >>
const def cd;

// CHECK: hl.var @vd, <external> : !hl.lvalue<!hl.elaborated<!hl.typedef<@def>,  volatile >>
volatile def vd;

// CHECK: hl.var @cvd, <external> constant : !hl.lvalue<!hl.elaborated<!hl.typedef<@def>,  const, volatile >>
const volatile def cvd;
