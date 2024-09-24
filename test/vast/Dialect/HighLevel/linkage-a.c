// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @foo internal
static void foo(void) {}

// CHECK: hl.func @bar available_externally
inline void bar(void) {}

// CHECK: hl.func @baz available_externally
inline void baz(void);
inline void baz(void) {}

// CHECK: hl.func @qux available_externally
inline void qux(void) {}
inline void qux(void);

// CHECK: hl.func @quux available_externally
inline void quux(void) {}
void quux(void);

// CHECK: hl.func @corge external
inline extern void corge(void);
inline void corge(void) {}

// CHECK: hl.func @grault available_externally
inline void grault(void) {}
inline extern void grault(void);

// CHECK: hl.func @graply internal
static inline void graply(void) {}

// CHECK: hl.func @waldo external
inline void waldo(void);
void waldo(void);
inline void waldo(void) {}

// CHECK: hl.func @fred external
inline void fred(void);
inline extern void fred(void);
inline void fred(void) {}

// CHECK: hl.func @plugh unknown{{.*}}{sym_visibility = "private"}
inline extern void plugh(void);
