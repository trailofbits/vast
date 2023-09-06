// RUN: printf "load %s\n raise vast-hl-to-ll-cf\n show module\n exit" | %vast-repl | FileCheck %s
// CHECK: ll.return %0 : !hl.int
int main(void) { return 0; }
