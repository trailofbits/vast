// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @loop_simple
void loop_simple()
{
    // CHECK: hl.for init
    // CHECK: cond
    // CHECK: incr
    // CHECK: body
    for (int i = 0; i < 100; i++) {}
}

// CHECK-LABEL: func private @loop_noinit
void loop_noinit()
{
    int i = 0;
    // CHECK: hl.for init
    // CHECK: cond
    // CHECK: incr
    // CHECK: body
    for (;i < 100; ++i) {}
}

// CHECK-LABEL: func private @loop_noincr
void loop_noincr()
{
    // CHECK: hl.for init
    // CHECK: cond
    // CHECK: incr
    // CHECK: body
    for (int i = 0; i < 100;) { ++i; }
}

// CHECK-LABEL: func private @loop_infinite
void loop_infinite()
{
    // CHECK: hl.for init
    // CHECK: cond
    // CHECK: incr
    // CHECK: body
    for (;;) {}
}

// CHECK-LABEL: func private @loop_nested
void loop_nested()
{
    // CHECK: hl.for init
    // CHECK: cond
    // CHECK: incr
    // CHECK: body
    for (int i = 0; i < 100; ++i) {
        // CHECK: hl.for init
        // CHECK: cond
        // CHECK: incr
        // CHECK: body
        for (int j = 0; j < 100; ++j) {

        }
    }
}
