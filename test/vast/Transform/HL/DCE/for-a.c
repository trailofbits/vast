// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-dce | FileCheck %s

void fn()
{
    for (int i = 0; i < 15; ++i)
    {
        // CHECK: } do {
        // CHECK-NEXT:   [[V1:%[0-9]+]] = hl.ref [[V0:%[0-9]+]] : !hl.lvalue<!hl.int>
        // CHECK-NEXT:   [[V2:%[0-9]+]] = hl.pre.inc [[V1]] : !hl.lvalue<!hl.int>
        // CHECK-NEXT:   hl.break
        // CHECK-NEXT: }
        ++i;
        break;
        --i;
        continue;
    }
}
