// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
// UNSUPPORTED: system-darwin

// CHECK-DAG: hl.typedef "thrd_t"
// CHECK-DAG: hl.func @thrd_create
// CHECK-DAG: hl.func @thrd_equal
// CHECK-DAG: hl.func @thrd_current
// CHECK-DAG: hl.func @thrd_sleep
// CHECK-DAG: hl.func @thrd_yield
// CHECK-DAG: hl.func @thrd_detach
// CHECK-DAG: hl.func @thrd_join

// CHECK-DAG: hl.enum.const "thrd_success"
// CHECK-DAG: hl.enum.const "thrd_busy"
// CHECK-DAG: hl.enum.const "thrd_error"
// CHECK-DAG: hl.enum.const "thrd_nomem"
// CHECK-DAG: hl.enum.const "thrd_timedout"

// CHECK-DAG: hl.typedef "mtx_t"
// CHECK-DAG: hl.func @mtx_init
// CHECK-DAG: hl.func @mtx_lock
// CHECK-DAG: hl.func @mtx_timedlock
// CHECK-DAG: hl.func @mtx_trylock
// CHECK-DAG: hl.func @mtx_unlock
// CHECK-DAG: hl.func @mtx_destroy

// CHECK-DAG: hl.enum.const "mtx_plain"
// CHECK-DAG: hl.enum.const "mtx_recursive"
// CHECK-DAG: hl.enum.const "mtx_timed"

// CHECK-DAG: hl.func @call_once

// CHECK-DAG: hl.typedef "cnd_t"
// CHECK-DAG: hl.func @cnd_init
// CHECK-DAG: hl.func @cnd_signal
// CHECK-DAG: hl.func @cnd_broadcast
// CHECK-DAG: hl.func @cnd_wait
// CHECK-DAG: hl.func @cnd_timedwait
// CHECK-DAG: hl.func @cnd_destroy

// CHECK-DAG: hl.typedef "tss_t"
// CHECK-DAG: hl.typedef "tss_dtor_t"
// CHECK-DAG: hl.func @tss_create
// CHECK-DAG: hl.func @tss_get
// CHECK-DAG: hl.func @tss_set
// CHECK-DAG: hl.func @tss_delete

#include <threads.h>
