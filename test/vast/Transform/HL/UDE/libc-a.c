// RUN: %vast-front -vast-emit-mlir=hl -vast-simplify %s -o - | %file-check %s
#include <stdio.h>

// CHECK-NOT: hl.typedef
// CHECK-NOT: hl.struct
// CHECK-NOT: hl.func
// CHECK-NOT: hl.var