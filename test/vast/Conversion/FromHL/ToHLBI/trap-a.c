// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %vast-opt --vast-hl-to-hl-builtin | %file-check %s
void fn() {
    // CHECK: hlbi.trap : !hl.void
    __builtin_trap();
}
