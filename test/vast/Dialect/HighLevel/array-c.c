// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @cai : !hl.lvalue<!hl.array<3, !hl.int< const >>> = {
// CHECK:   [[V1:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
// CHECK:   [[V2:%[0-9]+]] = hl.const #core.integer<2> : !hl.int
// CHECK:   [[V3:%[0-9]+]] = hl.const #core.integer<3> : !hl.int
// CHECK:   [[V4:%[0-9]+]] = hl.initlist [[V1]], [[V2]], [[V3]] : (!hl.int, !hl.int, !hl.int) -> !hl.array<3, !hl.int< const >>
// CHECK:   hl.value.yield [[V4]] : !hl.array<3, !hl.int< const >>
const int cai[3] = {1, 2, 3};
