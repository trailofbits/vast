// RUN: vast-front %s -vast-emit-mlir=hl -o - | FileCheck %s
// RUN: vast-front %s -vast-emit-mlir=hl -o - > %t && vast-opt %t | diff -B %t -
// REQUIRES: extern-in-namespace

// CHECK: unsup.decl "Namespace::N"
// CHECK:    unsup.decl "LinkageSpec"
// CHECK:       hl.func external @f ()
namespace N { extern "C" void f(); }
void N::f() {}
// CHECK: unsup.decl "Namespace::M"
// CHECK: unsup.decl "LinkageSpec"
// CHECK: hl.func external @f ()
namespace M { extern "C" void f(); }
