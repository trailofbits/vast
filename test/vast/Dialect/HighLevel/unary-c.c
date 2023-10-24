// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @logical
void logical(unsigned long a)
{
    // CHECK: hl.lnot [[V1:%[0-9]+]] : !hl.long< unsigned > -> !hl.int
    !a;
}

