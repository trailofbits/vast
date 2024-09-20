// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

typedef __builtin_va_list va_list;

#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)
#define va_copy(dst, src)   __builtin_va_copy(dst, src)

// CHECK: hl.typedef @__builtin_va_list
// CHECK: hl.typedef @va_list : !hl.elaborated<!hl.typedef<@__builtin_va_list>>

int average(int count, ...) {
// CHECK: hl.func @{{.*}}average{{.*}}(%arg0: !hl.lvalue<!hl.int>, ...) -> !hl.int
    va_list args, args_copy;
    va_start(args, count);
    // CHECK: hl.call @__builtin_va_start

    va_copy(args_copy, args);
    // CHECK: hl.call @__builtin_va_copy

    int sum = 0;
    for(int i = 0; i < count; i++) {
        sum += va_arg(args, int);
        // CHECK: hl.va_arg_expr
    }

    va_end(args);
    // CHECK: hl.call @__builtin_va_end

    return count > 0 ? sum / count : 0;
}

int test(void) {
  return average(5, 1, 2, 3, 4, 5);
  // CHECK: hl.call @average
}
