// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func @main() -> !highlevel.int
int main() {}
