// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-to-ll-func | %vast-opt

// Just check that we can re-load the low level function.
void foo(void) {}
