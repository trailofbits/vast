// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -
// REQUIRES: qualifiers

// CHECK: hl.var "cai" : !hl.lvalue<!hl.array<3, !hl.int<const>>> = {
// CHECK:   [[V1:%[0-9]+]] = hl.constant.int 1 : !hl.int
// CHECK:   [[V2:%[0-9]+]] = hl.constant.int 2 : !hl.int
// CHECK:   [[V3:%[0-9]+]] = hl.constant.int 3 : !hl.int
// CHECK:   [[V4:%[0-9]+]] = hl.initlist [[V1]], [[V2]], [[V3]] : (!hl.int, !hl.int, !hl.int) -> !hl.array<3, !hl.int<const>>
// CHECK:   hl.value.yield [[V4]] : !hl.array<3, !hl.int<const>>
const int cai[3] = {1, 2, 3};
