// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-emit-abi | %file-check %s -check-prefix=ABI

struct wrapped
{
    int x;
};

// ABI:      abi.func {{.*}} (%arg0: i32) -> si32 {{.*}}
// ABI-NEXT:   {{.*}} = abi.prologue {
// ABI-NEXT:     [[V5:%[0-9]+]] = abi.direct %arg0 : i32 -> !hl.lvalue<!hl.elaborated<!hl.record<@wrapped>>>
// ABI-NEXT:     [[V6:%[0-9]+]] = abi.yield [[V5]] : !hl.lvalue<!hl.elaborated<!hl.record<@wrapped>>> -> !hl.lvalue<!hl.elaborated<!hl.record<@wrapped>>>
int fn( struct wrapped w )
{
    // ABI:      [[V4:%[0-9]+]] = abi.epilogue {
    // ABI-NEXT:   [[V5:%[0-9]+]] = abi.direct {{.*}} : si32 -> si32
    // ABI-NEXT:   {{.*}} = abi.yield [[V5]] : si32 -> si32
    // ABI-NEXT: } : si32
    // ABI-NEXT: hl.return [[V4]] : si32
    return w.x;
}
