// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -
// REQUIRES: extern-in-namespace

// CHECK: unsup.decl "Namespace::N"
// CHECK:    unsup.decl "LinkageSpec"
// CHECK:       hl.func @f ()
namespace N { extern "C" void f(); }
void N::f() {}
// CHECK: unsup.decl "Namespace::M"
// CHECK: unsup.decl "LinkageSpec"
// CHECK: hl.func @f ()
namespace M { extern "C" void f(); }
