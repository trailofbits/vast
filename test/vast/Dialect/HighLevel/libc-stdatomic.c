// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.typedef "atomic_bool" : !hl.atomic<!hl.bool>
// CHECK-DAG: hl.typedef "atomic_char" : !hl.atomic<!hl.char>
// CHECK-DAG: hl.typedef "atomic_schar" : !hl.atomic<!hl.char>
// CHECK-DAG: hl.typedef "atomic_uchar" : !hl.atomic<!hl.char< unsigned >>
// CHECK-DAG: hl.typedef "atomic_short" : !hl.atomic<!hl.short>
// CHECK-DAG: hl.typedef "atomic_ushort" : !hl.atomic<!hl.short< unsigned >>
// CHECK-DAG: hl.typedef "atomic_int" : !hl.atomic<!hl.int>
// CHECK-DAG: hl.typedef "atomic_uint" : !hl.atomic<!hl.int< unsigned >>
// CHECK-DAG: hl.typedef "atomic_long" : !hl.atomic<!hl.long>
// CHECK-DAG: hl.typedef "atomic_ulong" : !hl.atomic<!hl.long< unsigned >>
// CHECK-DAG: hl.typedef "atomic_llong" : !hl.atomic<!hl.longlong>
// CHECK-DAG: hl.typedef "atomic_ullong" : !hl.atomic<!hl.longlong< unsigned >>
// CHECK-DAG: hl.typedef "atomic_char16_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_least16_t">>>
// CHECK-DAG: hl.typedef "atomic_char32_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_least32_t">>>
// CHECK-DAG: hl.typedef "atomic_wchar_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"wchar_t">>>
// CHECK-DAG: hl.typedef "atomic_int_least8_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_least8_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_least8_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_least8_t">>>
// CHECK-DAG: hl.typedef "atomic_int_least16_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_least16_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_least16_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_least16_t">>>
// CHECK-DAG: hl.typedef "atomic_int_least32_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_least32_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_least32_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_least32_t">>>
// CHECK-DAG: hl.typedef "atomic_int_least64_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_least64_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_least64_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_least64_t">>>
// CHECK-DAG: hl.typedef "atomic_int_fast8_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_fast8_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_fast8_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_fast8_t">>>
// CHECK-DAG: hl.typedef "atomic_int_fast16_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_fast16_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_fast16_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_fast16_t">>>
// CHECK-DAG: hl.typedef "atomic_int_fast32_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_fast32_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_fast32_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_fast32_t">>>
// CHECK-DAG: hl.typedef "atomic_int_fast64_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"int_fast64_t">>>
// CHECK-DAG: hl.typedef "atomic_uint_fast64_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uint_fast64_t">>>
// CHECK-DAG: hl.typedef "atomic_intptr_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"intptr_t">>>
// CHECK-DAG: hl.typedef "atomic_uintptr_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uintptr_t">>>
// CHECK-DAG: hl.typedef "atomic_size_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"size_t">>>
// CHECK-DAG: hl.typedef "atomic_ptrdiff_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"ptrdiff_t">>>
// CHECK-DAG: hl.typedef "atomic_intmax_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"intmax_t">>>
// CHECK-DAG: hl.typedef "atomic_uintmax_t" : !hl.atomic<!hl.elaborated<!hl.typedef<"uintmax_t">>>
#include <stdatomic.h>