// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: func private @while_break
void while_break()
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.constant true
    // CHECK: hl.cond.yield [[V1]]
    while (true) {
        // CHECK: } do {
        // CHECK: hl.break
        break;
    }
    // CHECK: }
}

// CHECK-LABEL: func private @while_continue
void while_continue()
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.constant true
    // CHECK: hl.cond.yield [[V1]]
    while (true) {
        // CHECK: } do {
        // CHECK: hl.continue
        continue;
    }
    // CHECK: }
}
