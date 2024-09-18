// RUN: %vast-cc1 -vast-emit-mlir-after=vast-strip-param-lvalues %s -o - | %file-check %s

// No new operations should be emitted since the args are not used.
// CHECK:  ll.func {{.*}}
// CHECK-NEXT: ll.cell @arg0
// CHECK-NEXT: ll.cell_init
// CHECK-NEXT: ll.cell @arg1
// CHECK-NEXT: ll.cell_init
// CHECK-NEXT: hl.const #void_value
void empty(int arg0, int arg1) {}
