// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int frob(int x)
{
  int y;

  // CHECK: hl.asm {has_goto} "frob %%r5, %1; jc %l[error]; mov (%2), %%r5"() (ins : ["fst", 1] {{.*}}, {{.*}} : ["r", "r"]) (["rax", "memory"]) ({{.*}}) : (!hl.int, !hl.ptr<!hl.int>, !hl.ptr<!hl.void>) -> ()
  asm goto ("frob %%r5, %1; jc %l[error]; mov (%2), %%r5"
            : /* No outputs. */
            : [fst] "r"(x), "r"(&y)
            : "rax", "memory"
            : error);
  return y;
error:
  return -1;
}
