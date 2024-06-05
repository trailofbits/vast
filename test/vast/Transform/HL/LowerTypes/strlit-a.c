// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s

int main()
{
    // CHECK: [[V2:%[0-9]+]] = hl.const "hello" : !hl.lvalue<!hl.array<6, si8>>
    const char *hello = "hello";
}
