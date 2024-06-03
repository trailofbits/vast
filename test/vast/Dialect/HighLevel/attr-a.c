// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.typedef "u64" : !hl.longlong< unsigned >
typedef unsigned long long u64;

# define __nocast	__attribute__((annotate("nocast")))

// CHECK: hl.typedef "cputime_t" {hl.annotation = #hl.annotation<"nocast">} : !hl.elaborated<!hl.typedef<"u64">>
typedef u64 __nocast cputime_t;
