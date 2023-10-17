// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @fun weak () -> !hl.void {
void __attribute__((__weak__)) fun (void) {}
int mian() {
    fun();
    return 0;
}
