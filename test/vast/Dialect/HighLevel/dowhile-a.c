// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @basic
void basic() {
    // CHECK: hl.do {
    do {
    } while (true);
    // CHECK: } while {
    // CHECK: [[V1:%[0-9]+]] = hl.constant( true )
    // CHECK: hl.cond.yield [[V1]]
    // CHECK: }
}

// CHECK-LABEL: func private @inner_cond
void inner_cond() {
    int i = 0;
    // CHECK: hl.do {
    // CHECK: [[V1:%[0-9]+]] = hl.declref( @i )
    // CHECK: hl.post.inc [[V1]]
    do {
        i++;
    } while (i < 100);
    // CHECK: } while {
    // CHECK: [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V2]]
    // CHECK: }
}
