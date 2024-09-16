// RUN: %vast-cc1 -vast-emit-mlir-after=vast-strip-param-lvalues %s -o - | %file-check %s

// No new operations should be emitted since the args are not used.
// CHECK:  ll.func {{.*}}
// CHECK-NEXT:    %0 = hl.const #void_value
void empty(int arg0, int arg1) {}
