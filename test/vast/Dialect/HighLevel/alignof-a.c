// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.alignof.type !hl.int -> !hl.long< unsigned >
    unsigned long ai = _Alignof(int);
}
