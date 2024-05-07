// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

int main() {
// CHECK: hl.var "k" : !hl.lvalue<!hl.typeof.type<!hl.int>>
typeof(int) k = 0;
return 0;
}
