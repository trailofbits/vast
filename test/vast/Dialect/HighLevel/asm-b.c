// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void f() {
int src = 1;
// CHECK: hl.var @src
int dst;
// CHECK: hl.var @dst
// CHECK: [[DST_REF:%[0-9]+]] = hl.ref @dst
// CHECK: [[SRC_REF:%[0-9]+]] = hl.ref @src
// CHECK: [[CAST_SRC:%[0-9]+]] = hl.implicit_cast [[SRC_REF]]
// CHECK: hl.asm {is_volatile} "mov %w1, %w0\0A\09add $1, %w0"([0] [[DST_REF]] : ["=r"]) (ins : [1] [[CAST_SRC]] : ["r"]) () () : (!hl.lvalue<!hl.int>, !hl.int) -> ()
asm inline volatile ("mov %w1, %w0\n\t"
    "add $1, %w0"
    : "=r" (dst)
    : "r" (src));
}
