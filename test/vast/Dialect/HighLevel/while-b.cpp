// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z11while_breakv
void while_break()
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.const #true
    // CHECK: hl.cond.yield [[V1]]
    while (true) {
        // CHECK: } do {
        // CHECK: hl.break
        break;
    }
    // CHECK: }
}

// CHECK-LABEL: hl.func @_Z14while_continuev
void while_continue()
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.const #true
    // CHECK: hl.cond.yield [[V1]]
    while (true) {
        // CHECK: } do {
        // CHECK: hl.continue
        continue;
    }
    // CHECK: }
}
