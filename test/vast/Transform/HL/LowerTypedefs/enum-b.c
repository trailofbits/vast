// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-lower-elaborated-types --vast-hl-lower-typedefs | %file-check %s


typedef enum { A_fst = 0 } A;
typedef enum { B_fst = 0 } B;

// CHECK: hl.func @foo {{.*}} (!hl.lvalue<!hl.enum<@"anonymous[{{.*}}]">>, !hl.lvalue<!hl.enum<@"anonymous[{{.*}}]">>) -> none attributes {sym_visibility = "private"}
void foo(A a, B b);
