// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %vast-opt --vast-hl-to-hl-builtin | %file-check %s
void fn() {
    // CHECK: hlbi.trap
    __builtin_trap();
}
