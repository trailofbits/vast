// RUN: vast-front %s -vast-emit-mlir=hl -o - | FileCheck %s
// RUN: vast-front %s -vast-emit-mlir=hl -o - > %t && vast-opt %t | diff -B %t -

// CHECK: unsupported.decl "StaticAssert"
static_assert(1, "Test static assert 1");
