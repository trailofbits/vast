// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.gnu_inline = #hl.gnu_inline
__attribute__((gnu_inline))
inline void fn(void) {
// CHECK: hl.unused = #hl.unused
    int x __attribute__((unused)) = 0;
}

// CHECK: hl.used = #hl.used
__attribute__((used))
void fn3(void) {}
