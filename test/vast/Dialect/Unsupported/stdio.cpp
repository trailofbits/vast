// RUN: vast-front %s -vast-emit-mlir=hl -o - | FileCheck %s
// RUN: vast-front %s -vast-emit-mlir=hl -o - > %t && vast-opt %t | diff -B %t -

// CHECK: unsupported.decl "LinkageSpec"
#include <stdio.h>
