// RUN: %vast-cc --from-source %s | FileCheck %s
// RUN: %vast-cc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z7logicalm
void logical(unsigned long a)
{
    // CHECK: hl.lnot [[V1:%[0-9]+]] : !hl.bool -> !hl.bool
    !a;
}

