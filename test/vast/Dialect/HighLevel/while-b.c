// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @while_break
void while_break()
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.constant( true )
    // CHECK: hl.cond.yield [[V1]]
    while (true) {
        // CHECK: } do {
        // CHECK: hl.break
        break;
    }
    // CHECK: }
}
