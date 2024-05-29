// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @fun external () -> !hl.void attributes {deprecated = #hl.deprecated<msg : "msg", fix : "fix">} {
__attribute__((deprecated("msg","fix")))
void fun (void) {};
// CHECK: hl.func @fun1 external () -> !hl.void attributes {deprecated = #hl.deprecated<msg : "msg", fix : "">} {
__attribute__((deprecated("msg")))
void fun1 (void) {};
// CHECK: hl.func @fun2 external () -> !hl.void attributes {deprecated = #hl.deprecated<msg : "", fix : "">} {
__attribute__((deprecated()))
void fun2 (void) {};
