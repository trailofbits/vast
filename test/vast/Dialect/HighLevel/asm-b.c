// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void f() {
int src = 1;
int dst;
// CHECK: hl.asm {is_volatile} "mov %1, %0\0A\09add $1, %0"([0] %2 : ["=r"]) (ins : [1] %4 : ["r"]) () () : (!hl.lvalue<!hl.int>, !hl.int) -> ()
asm inline volatile ("mov %1, %0\n\t"
    "add $1, %0"
    : "=r" (dst)
    : "r" (src));
}
