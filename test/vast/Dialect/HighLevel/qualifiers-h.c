// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.enum "e" : !hl.int< unsigned >
enum e { a, b, c };

// CHECK: hl.var "v" : !hl.lvalue<!hl.elaborated<!hl.record<"e">>>
enum e v;

// CHECK: hl.var "cv" : !hl.lvalue<!hl.elaborated<!hl.record<"e">,  const >>
const enum e cv;

// CHECK: hl.var "cvv" : !hl.lvalue<!hl.elaborated<!hl.record<"e">,  const, volatile >>
const volatile enum e cvv;

// CHECK: hl.typedef "def" : !hl.elaborated<!hl.record<"e">>
typedef enum e def;

// CHECK: hl.var "d" : !hl.lvalue<!hl.typedef<"def">>
def d;

// CHECK: hl.var "cd" : !hl.lvalue<!hl.typedef<"def",  const >>
const def cd;

// CHECK: hl.var "vd" : !hl.lvalue<!hl.typedef<"def",  volatile >>
volatile def vd;

// CHECK: hl.var "cvd" : !hl.lvalue<!hl.typedef<"def",  const, volatile >>
const volatile def cvd;
