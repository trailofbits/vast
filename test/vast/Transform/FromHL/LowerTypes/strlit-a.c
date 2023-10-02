// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-lower-types | %file-check %s

// CHECK-LABEL: hl.func @main () -> si32
int main()
{
    // CHECK: [[V2:%[0-9]+]] = hl.const #core.strlit<"hello"> : !hl.lvalue<memref<6xsi8>>
    const char *hello = "hello";
}
