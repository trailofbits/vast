// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

void add(int* a, int b) {
    // CHECK: [[A1:%[0-9]+]] = hl.add [[B1:%[0-9]+]], [[C1:%[0-9]+]] : (!hl.ptr<!hl.int>, !hl.int) -> !hl.ptr<!hl.int>
    int* res1 = a + b;
    // CHECK: [[A2:%[0-9]+]] = hl.add [[B2:%[0-9]+]], [[C2:%[0-9]+]] : (!hl.int, !hl.ptr<!hl.int>) -> !hl.ptr<!hl.int>
    int* res2 = b + a;
    // CHECK: [[A3:%[0-9]+]] = hl.sub [[B3:%[0-9]+]], [[C3:%[0-9]+]] : (!hl.ptr<!hl.int>, !hl.ptr<!hl.int>) -> !hl.long
    int res3 = a - a;
    // CHECK: [[A4:%[0-9]+]] = hl.sub [[B4:%[0-9]+]], [[C4:%[0-9]+]] : (!hl.ptr<!hl.int>, !hl.int) -> !hl.ptr<!hl.int>
    int* res4 = a - b;
    // CHECK: [[A5:%[0-9]+]] = hl.assign.add [[B5:%[0-9]+]] to [[C5:%[0-9]+]] : !hl.int, !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    a += b;
    // CHECK: [[A6:%[0-9]+]] = hl.assign [[B6:%[0-9]+]] to  [[C6:%[0-9]+]] : !hl.ptr<!hl.int>, !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    a = &b;
    // CHECK: [[A7:%[0-9]+]] = hl.assign.add [[B7:%[0-9]+]] to [[C7:%[0-9]+]] : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
    b += b;
    // CHECK: [[A8:%[0-9]+]] = hl.assign.sub [[B8:%[0-9]+]] to [[C8:%[0-9]+]] : !hl.int, !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    a -= b;
}
