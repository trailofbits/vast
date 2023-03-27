// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK: hl.func external @abort
// CHECK-NOT: hl.func external @abort
// CHECK: hl.func external @main

void abort();
void abort();
int main() {}
