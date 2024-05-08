// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -

#define __BEGIN_DECLS   extern "C" {
#define __END_DECLS     }

// CHECK: unsup.decl "LinkageSpec"
__BEGIN_DECLS
typedef signed char __int8_t;
typedef short __int16_t;
typedef unsigned short __uint16_t;
typedef int __int32_t;
typedef unsigned int __uint32_t;
typedef long long __int64_t;
typedef unsigned long long __uint64_t;
__END_DECLS
