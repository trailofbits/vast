// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.typedef "array_t" : !hl.array<10, !hl.int>
typedef int array_t[10];
// CHECK: hl.typedef "array_ptr_t" : !hl.ptr<!hl.elaborated<!hl.typedef<"array_t">>>
typedef array_t* array_ptr_t;

// CHECK: hl.func @foo {{.*}} (%arg0: !hl.lvalue<!hl.elaborated<!hl.typedef<"array_ptr_t">>>)
void foo(array_ptr_t array_ptr) {
    int x = (*array_ptr)[1];
}

void bar() {
    // CHECK: hl.var "arr_10" : !hl.lvalue<!hl.array<10, !hl.int>>
    int arr_10[10];
    // CHECK: hl.call @foo([[V:%[0-9]+]]) : (!hl.ptr<!hl.array<10, !hl.int>>)
    foo(&arr_10);
}
