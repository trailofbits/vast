// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// tag naming an unknown struct declares it
// CHECK: hl.type.decl @struct.s
// CHECK: hl.global @p : !hl.ptr<!hl.named_type<@struct.s>> = {
// CHECK:  [[V1:%[0-9]+]] = hl.constant.int 0 : !hl.int
// CHECK:  [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] NullToPointer : !hl.int -> !hl.ptr<!hl.named_type<@struct.s>>
// CHECK:  hl.value.yield [[V2]] : !hl.ptr<!hl.named_type<@struct.s>>
// CHECK: }
struct s* p = 0;

// definition for the struct pointed to by p
// CHECK: hl.record @struct.s : {
// CHECK:  hl.field @a : !hl.int
// CHECK: }
struct s { int a; };

// CHECK: func @g() -> !hl.void
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
