// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK: hl.func external @abort
// CHECK-NOT: hl.func external @abort
// CHECK: hl.func external @main
typedef void (abort_t)();
abort_t abort;
void abort();
int main() {}
