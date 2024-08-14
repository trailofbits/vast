// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.typedef @function_pointer_t : !hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>>>
typedef void (*function_pointer_t)(int); // <-- this creates a function pointer type
// CHECK: hl.typedef @function_t : !core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>
typedef void function_t(int); // <-- this creates a function type

// CHECK: hl.func @my_func {{.*}} (!hl.lvalue<!hl.int>)
function_t my_func; // <-- this declares "void my_func(int)"

// CHECK: hl.func @bar
void bar() {
    // CHECK: hl.call @my_func
    my_func(42);
}
