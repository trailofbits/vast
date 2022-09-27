// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: @chptr(!hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.void
void chptr(char *p);

// CHECK: @vptr(!hl.lvalue<!hl.ptr<!hl.void>>) -> !hl.void
void vptr(void *p);

// CHECK: @cchptr(!hl.lvalue<!hl.ptr<!hl.char< const >>>) -> !hl.void
void cchptr(const char *p);

// CHECK: @cvptr(!hl.lvalue<!hl.ptr<!hl.void< const >>>) -> !hl.void
void cvptr(const void *p);

// CHECK: @cchcptr(!hl.lvalue<!hl.ptr<!hl.char< const >,  const >>) -> !hl.void
void cchcptr(const char * const p);

// CHECK: @cvcptr(!hl.lvalue<!hl.ptr<!hl.void< const >,  const >>) -> !hl.void
void cvcptr(const void * const p);

// CHECK: @cvchptr(!hl.lvalue<!hl.ptr<!hl.char< const, volatile >>>) -> !hl.void
void cvchptr(const volatile char *p);

// CHECK: @cvvptr(!hl.lvalue<!hl.ptr<!hl.void< const, volatile >>>) -> !hl.void
void cvvptr(const volatile void *p);

// CHECK: @cvrchptr(!hl.lvalue<!hl.ptr<!hl.char< const, volatile >,  restrict >>) -> !hl.void
void cvrchptr(const volatile char * restrict p);

// CHECK: @cvrvptr(!hl.lvalue<!hl.ptr<!hl.void< const, volatile >,  restrict >>) -> !hl.void
void cvrvptr(const volatile void * restrict p);

// CHECK: @cvvcvrptr(!hl.lvalue<!hl.ptr<!hl.void< const, volatile >,  const, volatile, restrict >>) -> !hl.void
void cvvcvrptr(const volatile void * const volatile restrict ptr);

