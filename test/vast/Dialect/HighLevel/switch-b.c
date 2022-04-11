// RUN: vast-cc --ccopts -std=c++17 --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -std=c++17 --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: func @switch_init([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int switch_init(int num)
{
    // CHECK: hl.scope {
    // CHECK:   [[V:%[0-9]+]] = hl.var "v" : !hl.lvalue<!hl.int>
    // CHECK:   hl.switch {
    // CHECK:       [[V2:%[0-9]+]] = hl.declref [[V]]
    // CHECK:       [[V3:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK:       hl.value.yield [[V3]]
    switch (int v = num; v) {
        // CHECK: } cases {
        case  1: return 1;
        case  2: return 2;
        default: return 0;
    }
    // CHECK: hl.var "x" : !hl.lvalue<!hl.int>
    int x = 0;
    return x;
}
