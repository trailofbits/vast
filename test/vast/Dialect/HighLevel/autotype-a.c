// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#define auto_t(a) \
  ({ __auto_type _a = (a); _a; })

#define auto_tc(a) \
  ({ const __auto_type _a = (a); _a; })

int main() {
    int x = 0;
    //CHECK: hl.var @_a : !hl.lvalue<!hl.auto<!hl.int>>
    int y = auto_t(x);
    //CHECK: hl.var @_a constant : !hl.lvalue<!hl.auto<!hl.int,  const >>
    int z = auto_tc(x);
    //CHECK: hl.var @u : !hl.lvalue<!hl.auto<!hl.int>>
    __auto_type u = z;
}
