// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var "earr" sc_extern : !hl.lvalue<!hl.array<?, !hl.int>>
extern int earr[];
