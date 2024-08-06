// RUN: %vast-front -vast-emit-mlir=hl -fcf-protection %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -fcf-protection %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.gnu_inline = #hl.gnu_inline
__attribute__((gnu_inline))
inline void fn(void) {
// CHECK: hl.unused = #hl.unused
    int x __attribute__((unused)) = 0;
}

// CHECK: hl.nocf_check = #hl.nocf_check
__attribute__((nocf_check))
void fn2(void) {}

// CHECK: hl.used = #hl.used
__attribute__((used))
void fn3(void) {}
