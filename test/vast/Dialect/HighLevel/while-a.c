// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @while_empty
void while_empty()
{
    // CHECK: [[V1:%[0-9]+]] = hl.constant( true )
    // CHECK-NEXT: hl.while [[V1]]
    while (true)
    {
        // CHECK: hl.scope.end
    }
    // CHECK: hl.scope.end
}

// CHECK-LABEL: func private @while_simple
void while_simple(int a)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sgt
    // CHECK-NEXT: hl.while [[V1]]
    while (a > 0)
    {
        a = a - 1;
        // CHECK: [[V2:%[0-9]+]] = hl.sub
        // CHECK: hl.assign [[V2]]
        // CHECK: hl.scope.end
    }
    // CHECK: hl.scope.end
}

// CHECK-LABEL: func private @while_nested
void while_nested(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sgt
    // CHECK-NEXT: hl.while [[V1]]
    while (a > 0) {
        int c = b;
        // CHECK: [[V2:%[0-9]+]] = hl.cmp sgt
        // CHECK-NEXT: hl.while [[V2]]
        while (c > 0) {
            c = c - 1;
            // CHECK: hl.scope.end
        }
        a = a - 1;
        // CHECK: hl.scope.end
    }
    // CHECK: hl.scope.end
}
