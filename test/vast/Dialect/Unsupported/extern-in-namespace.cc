// RUN: vast-front %s -vast-emit-mlir=hl -o - | FileCheck %s
// RUN: vast-front %s -vast-emit-mlir=hl -o - > %t && vast-opt %t | diff -B %t -
// REQUIRES: extern-in-namespace

// CHECK: unsupported.decl "Namespace::N"
// CHECK:    unsupported.decl "LinkageSpec"
// CHECK:       hl.func external @f ()
namespace N { extern "C" void f(); }
void N::f() {}
// CHECK: unsupported.decl "Namespace::M"
// CHECK: unsupported.decl "LinkageSpec"
// CHECK: hl.func external @f ()
namespace M { extern "C" void f(); }
