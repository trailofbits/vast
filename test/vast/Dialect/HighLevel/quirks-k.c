// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.var "foo" sc_extern : !hl.lvalue<!hl.void>
extern void foo;
