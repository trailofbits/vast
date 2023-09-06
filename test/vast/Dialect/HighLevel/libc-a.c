// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -
// REQUIRES: libc

#include <stdio.h>

int main() {
    printf("hello world\n");
}
