// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

typedef int INT;
int main() {
    // CHECK: hl.builtin_types_compatible_p.type !hl.int, !hl.char compatible false -> !hl.int
    __builtin_types_compatible_p(int, char);
    // CHECK: hl.builtin_types_compatible_p.type !hl.int, !hl.int compatible true -> !hl.int
    __builtin_types_compatible_p(int, int);
    // CHECK: hl.builtin_types_compatible_p.type !hl.int, !hl.elaborated<!hl.typedef<"INT">> compatible true -> !hl.int
    __builtin_types_compatible_p(int, INT);
}
