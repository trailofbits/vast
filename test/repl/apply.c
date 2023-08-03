// RUN: printf "load %s\n apply\n exit" | vast-repl | FileCheck %s
// CHECK: ll.return %0 : !hl.int
int main(void) { return 0; }
