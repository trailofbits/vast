// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int frob(int x)
{
  int y;

  // CHECK: hl.asm {has_goto} #core.strlit<"frob %%r5, %1; jc %l[error]; mov (%2), %%r5">() (ins : [#core.strlit<"fst">, 1] %3, %5 : [#core.strlit<"r">, #core.strlit<"r">]) ([#core.strlit<"rax">, #core.strlit<"memory">]) (%6) : (!hl.int, !hl.ptr<!hl.int>, !hl.lvalue<!hl.ptr<!hl.void>>) -> ()
  asm goto ("frob %%r5, %1; jc %l[error]; mov (%2), %%r5"
            : /* No outputs. */
            : [fst] "r"(x), "r"(&y)
            : "rax", "memory"
            : error);
  return y;
error:
  return -1;
}
