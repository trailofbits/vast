// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl -vast-show-locs %s -o - | %file-check %s -check-prefix=LOCS
// RUN: %vast-cc1 -vast-emit-mlir=hl -vast-show-locs -vast-locs-as-meta-ids %s -o - | %file-check %s -check-prefix=META
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {}

// CHECK: module @"{{.*}}vast/Dialect/HighLevel/main-a.c"
// CHECK: hl.func @main {{.*}} () -> !hl.int

// LOCS: hl.func @main {{.*}} () -> !hl.int {
// LOCS: } {{.*}}/main-a.c:6:5
// LOCS: } {{.*}}/main-a.c:0:0

// META: hl.func @main {{.*}} () -> !hl.int {
// META: } <#meta.id<0>>
