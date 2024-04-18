// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: @chptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.char>>)
void chptr(char *p);

// CHECK: @vptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.void>>)
void vptr(void *p);

// CHECK: @cchptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.char< const >>>)
void cchptr(const char *p);

// CHECK: @cvptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.void< const >>>)
void cvptr(const void *p);

// CHECK: @cchcptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.char< const >,  const >>)
void cchcptr(const char * const p);

// CHECK: @cvcptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.void< const >,  const >>)
void cvcptr(const void * const p);

// CHECK: @cvchptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.char< const, volatile >>>)
void cvchptr(const volatile char *p);

// CHECK: @cvvptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.void< const, volatile >>>)
void cvvptr(const volatile void *p);

// CHECK: @cvrchptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.char< const, volatile >,  restrict >>)
void cvrchptr(const volatile char * restrict p);

// CHECK: @cvrvptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.void< const, volatile >,  restrict >>)
void cvrvptr(const volatile void * restrict p);

// CHECK: @cvvcvrptr {{.*}} (!hl.lvalue<!hl.ptr<!hl.void< const, volatile >,  const, volatile, restrict >>)
void cvvcvrptr(const volatile void * const volatile restrict ptr);
