// RUN: printf "load %s\n raise vast-hl-to-ll-cf\n show module\n exit" | %vast-repl | %file-check %s
// CHECK: hl.return %0 : !hl.int

// REQUIRES: clone-memory-leak

int main(void) { return 0; }
