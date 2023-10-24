// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void logic_assign_to_different_type() {
    // CHECK: hl.bin.lor {
    // CHECK: hl.value.yield [[A:%[0-9]+]] : !hl.long< unsigned >
    // CHECK: }, {
    // CHECK: hl.value.yield [[B:%[0-9]+]] : !hl.int
    // CHECK: } : !hl.int
    int a = (+1UL) || 0;
}
