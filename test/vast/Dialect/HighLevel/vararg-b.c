// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple aarch64-none-linux-android24 -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef __builtin_va_list va_list;

#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)
#define va_copy(dst, src)   __builtin_va_copy(dst, src)

// CHECK: hl.typedef "va_list" : !hl.elaborated<!hl.typedef<"__builtin_va_list">>

int average(int count, ...) {
// CHECK: cir.func @{{.*}}average{{.*}}(%arg0: !hl.lvalue<!hl.int>, ...) -> !hl.int
    va_list args, args_copy;
    va_start(args, count);
    // CHECK: hl.call @__builtin_va_start

    va_copy(args_copy, args);
    // CHECK: hl.call @__builtin_va_copy

    int sum = 0;
    for(int i = 0; i < count; i++) {
        sum += va_arg(args, int);
        // CHECK: VAArgExpr
    }

    va_end(args);
    // CHECK: hl.call @__builtin_va_end

    return count > 0 ? sum / count : 0;
}

int test(void) {
  return average(5, 1, 2, 3, 4, 5);
  // CHECK: hl.call @average
}
