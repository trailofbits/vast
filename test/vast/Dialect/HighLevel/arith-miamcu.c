// RUN: %vast-front -m32 -miamcu -vast-emit-mlir=hl %S/add-a.c -o - | %file-check %S/add-a.c
// RUN: %vast-front -m32 -miamcu -vast-emit-mlir=hl %S/sub-a.c -o - | %file-check %S/sub-a.c
// REQUIRES: miamcu
