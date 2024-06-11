// RUN: %vast-cc1 -O0 -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=O0
// RUN: %vast-cc1 -O1 -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=O1
// RUN: %vast-cc1 -O2 -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=O2
// RUN: %vast-cc1 -O3 -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=O3

// O0: hlbi.trap
// O1: hl.unreachable
// O2: hl.unreachable
// O3: hl.unreachable
int missing_return() {}