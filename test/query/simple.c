// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-query --show-symbols=functions %t | FileCheck %s

// CHECK: func : main
int main() {}
