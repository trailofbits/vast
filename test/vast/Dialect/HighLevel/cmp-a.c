// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

typedef unsigned long int uintptr_t;

#define NULL 0

int int_int_cmp(int a, int b) {
    // CHECK: hl.cmp slt [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.int, !hl.int -> !hl.int
    return a < b;
}

int cint_int_cmp(const int a, int b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.int, !hl.int -> !hl.int
    return a == b;
}

int uint_int_cmp(unsigned int a, int b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.int< unsigned >, !hl.int< unsigned > -> !hl.int
    return a == b;
}

int int_uint_cmp(int a, unsigned int b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.int< unsigned >, !hl.int< unsigned > -> !hl.int
    return a == b;
}

int vint_int_cmp(volatile int a, int b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.int, !hl.int -> !hl.int
    return a == b;
}

int char_con_cmp(char a) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.int, !hl.int -> !hl.int
    return a == 'b';
}

int int_long_cmp(int a, long b) {
    // CHECK: hl.cmp slt [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.long, !hl.long -> !hl.int
    return a < b;
}

int int_ptr_cmp(unsigned long a, void* b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.long< unsigned >, !hl.elaborated<!hl.typedef<@uintptr_t>> -> !hl.int
    return a == (uintptr_t)b;
}

int ptr_int_cmp(void *a, unsigned b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.elaborated<!hl.typedef<@uintptr_t>>, !hl.elaborated<!hl.typedef<@uintptr_t>> -> !hl.int
    return (uintptr_t)a == b;
}

int ptr_ptr_cmp(void *a, void *b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.ptr<!hl.void>, !hl.ptr<!hl.void> -> !hl.int
    return a == b;
}

int rptr_ptr_cmp(void * restrict a, void  *b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.ptr<!hl.void>, !hl.ptr<!hl.void> -> !hl.int
    return a == b;
}

int iptr_iptr_cmp(int *a, int *b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.ptr<!hl.int>, !hl.ptr<!hl.int> -> !hl.int
    return a == b;
}

int ptr_null_cmp(void *a) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.ptr<!hl.void>, !hl.ptr<!hl.void> -> !hl.int
    return a == NULL;
}

int iptr_null_cmp(int *a) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.ptr<!hl.int>, !hl.ptr<!hl.int> -> !hl.int
    return a == NULL;
}

int float_float_cmp(float a, float b) {
    // CHECK: hl.fcmp oeq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.float, !hl.float -> !hl.int
    return a == b;
}
