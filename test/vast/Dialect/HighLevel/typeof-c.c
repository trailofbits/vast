// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

int main() {
// CHECK: hl.var @i : !hl.lvalue<!hl.int>
int i = 0;

// CHECK: hl.typeof.expr "(i)" {
// CHECK:   hl.expr : !hl.lvalue<!hl.int> {
// CHECK:   }
// CHECK:   hl.type.yield {{.*}} : !hl.lvalue<!hl.int>
// CHECK: } : !hl.int
// CHECK: hl.var @j : !hl.lvalue<!hl.typeof.expr<"(i)", const >>
// CHECK: hl.const #core.integer<0> : !hl.int
const typeof(i) j = 0;
// CHECK: hl.var @k : !hl.lvalue<!hl.typeof.type<!hl.int, const >>
const typeof(int) k = 0;
return 0;
}
