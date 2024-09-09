// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-to-ll-geps | %file-check %s

struct X { int a; };

void fn()
{
    struct X x;
    // CHECK: [[V:%[0-9]+]] = "ll.gep"({{.*}}) <{field = @a, idx = 0 : i32}> : (!hl.lvalue<!hl.elaborated<!hl.record<"X">>>) -> !hl.lvalue<si32>
    x.a = 5;
}
