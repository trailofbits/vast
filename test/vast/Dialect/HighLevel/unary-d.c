// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func external @logical
void logical(unsigned long a)
{
    // CHECK: hl.lnot [[V1:%[0-9]+]] : !hl.bool -> !hl.bool
    !a;
}

