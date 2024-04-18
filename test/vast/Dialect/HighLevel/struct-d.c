// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.type "s"

// tag naming an unknown struct declares it
// CHECK: hl.var "p" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"s">>>> = {
// CHECK:  [[V1:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
// CHECK:  [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] NullToPointer : !hl.int -> !hl.ptr<!hl.elaborated<!hl.record<"s">>>

// CHECK:  hl.value.yield [[V2]] : !hl.ptr<!hl.elaborated<!hl.record<"s">>>
// CHECK: }
struct s* p = 0;

// definition for the struct pointed to by p
// CHECK: hl.struct "s" : {
// CHECK:  hl.field "a" : !hl.int
// CHECK: }
struct s { int a; };

// CHECK: hl.func @g
void g(void)
{
    // TODO: locally scoped structs

    // forward declaration of a new, local struct s
    // this hides global struct s until the end of this block
    struct s;
    // pointer to local struct s
    // without the forward declaration above,
    // this would point at the file-scope s
    struct s *p;
    // definitions of the local struct s
    struct s { char* p; };
}
