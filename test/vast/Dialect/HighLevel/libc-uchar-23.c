// RUN: %vast-front -vast-emit-mlir=hl -std=c23 %S/libc-uchar.c -o - | %file-check %S/libc-uchar.c --check-prefixes C23,CHECK
// REQUIRES: ucharc23
