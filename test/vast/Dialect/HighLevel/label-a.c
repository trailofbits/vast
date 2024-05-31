// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int c;

void foo (int x)
{
  static int b[] = { &&lab1 - &&lab0, &&lab2 - &&lab0 };
  // CHECK: hl.add {{.*}} : (!hl.ptr<!hl.void>, !hl.int) -> !hl.ptr<!hl.void>
  void *a = &&lab0 + b[x];
  goto *a;
lab1:
  c += 2;
lab2:
  c++;
lab0:
  ;
}
