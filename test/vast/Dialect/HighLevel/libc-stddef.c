// RUN: %vast-front -vast-emit-mlir=hl -std=c23 %s -o - | %file-check --check-prefix C23 %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check --check-prefix PRE-C23 %s
// RUN: %vast-front -vast-emit-mlir=hl -std=c23 %s -o %t && %vast-opt %t | diff -B %t -

// C23-DAG: hl.typedef "size_t"
// C23-DAG: hl.typedef "ptrdiff_t"
// C23-DAG: hl.typedef "max_align_t"
// C23-DAG: hl.typedef "nullptr_t"
// PRE-C23-NOT: hl.typedef "nullptr_t"
#include <stddef.h>
