// RUN: %vast-front -vast-emit-mlir-after=vast-emit-abi %s -o - | %file-check %s -check-prefix=ABI

struct wrapped
{
    int x;
};

// ABI:      abi.func {{.*}} (%arg0: i32) -> si32 {{.*}}
// ABI-NEXT:   {{.*}} = abi.prologue {
// ABI-NEXT:     [[V5:%[0-9]+]] = abi.direct %arg0 : i32 -> !hl.record<@wrapped>
// ABI-NEXT:     [[V6:%[0-9]+]] = abi.yield [[V5]] : !hl.record<@wrapped> -> !hl.record<@wrapped>
int fn( struct wrapped w )
{
    // ABI:      [[V4:%[0-9]+]] = abi.epilogue {
    // ABI-NEXT:   [[V5:%[0-9]+]] = abi.direct {{.*}} : si32 -> si32
    // ABI-NEXT:   {{.*}} = abi.yield [[V5]] : si32 -> si32
    // ABI-NEXT: } : si32
    // ABI-NEXT: ll.return [[V4]] : si32
    return w.x;
}
