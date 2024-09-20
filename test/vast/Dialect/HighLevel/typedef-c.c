// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.typedef @INT : !hl.int
typedef int INT;
void fun(INT a, int b) {
    // CHECK: hl.assign.add [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.int, !hl.lvalue<[[LHS_T:!hl.elaborated<!hl.typedef<@INT>>]]> -> [[LHS_T]]
    a += b;
    // CHECK: hl.assign.sub [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.int, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    a -= b;
    // CHECK: hl.assign.mul [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.int, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    a *= b;
    // CHECK: hl.assign.sdiv [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.int, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    a /= b;
    // CHECK: hl.assign.srem [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.int, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    a %= b;

    // CHECK: hl.assign.add [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT>>, !hl.lvalue<[[LHS_T:!hl.int]]> -> [[LHS_T]]
    b += a;
    // CHECK: hl.assign.sub [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT>>, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    b -= a;
    // CHECK: hl.assign.mul [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT>>, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    b *= a;
    // CHECK: hl.assign.sdiv [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT>>, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    b /= a;
    // CHECK: hl.assign.srem [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT>>, !hl.lvalue<[[LHS_T]]> -> [[LHS_T]]
    b %= a;

    INT c;
    // CHECK: hl.assign [[X:%[0-9]+]] to [[Y:%[0-9]+]] : [[TYPE:!hl.elaborated<!hl.typedef<@INT>>]], !hl.lvalue<[[TYPE]]> -> [[TYPE]]
    c = a;
    // CHECK: hl.assign [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.int, !hl.lvalue<[[LHS_T:!hl.elaborated<!hl.typedef<@INT>>]]> -> [[LHS_T]]
    c = b;
    // CHECK: hl.assign [[X:%[0-9]+]] to [[Y:%[0-9]+]] : !hl.elaborated<!hl.typedef<@INT>>, !hl.lvalue<[[LHS_T:!hl.int]]> -> [[LHS_T]]
    b = a;
}
