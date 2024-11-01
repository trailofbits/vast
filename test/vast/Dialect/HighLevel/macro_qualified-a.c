// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
#define CDECL __attribute__((cdecl))
// CHECK: !hl.ptr<!hl.macro_qualified<{{.*}}>>
typedef void (CDECL *X)();
