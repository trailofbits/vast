// RUN: %vast-cc --from-source %s | %file-check %s
// RUN: %vast-cc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z11while_breakv
void while_break()
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.const #hl.bool<true> : !hl.bool
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
    // CHECK: [[V1:%[0-9]+]] = hl.const #hl.bool<true> : !hl.bool
    // CHECK: hl.cond.yield [[V1]]
    while (true) {
        // CHECK: } do {
        // CHECK: hl.continue
        continue;
    }
    // CHECK: }
}
