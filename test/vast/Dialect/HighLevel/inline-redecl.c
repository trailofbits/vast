// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-NOT: hl.func @strlen{{.*}}hl.builtin
extern inline __attribute__((always_inline)) __attribute__((gnu_inline)) unsigned long strlen(const char *p) {
  return 1;
}
unsigned long mystrlen(char const *s) {
  return strlen(s);
}
unsigned long strlen(const char *s) {
  return 2;
}
unsigned long yourstrlen(char const *s) {
  return strlen(s);
}
