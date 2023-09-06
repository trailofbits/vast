// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.typedef "u64" : !hl.longlong< unsigned >
typedef unsigned long long u64;

# define __nocast	__attribute__((annotate("nocast")))

// CHECK: hl.typedef "cputime_t" {annotation = #hl<annotation "nocast">} : !hl.elaborated<!hl.typedef<"u64">>
typedef u64 __nocast cputime_t;
