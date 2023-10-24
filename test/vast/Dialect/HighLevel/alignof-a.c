// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.alignof.type !hl.int -> !hl.long< unsigned >
    unsigned long ai = _Alignof(int);
}
