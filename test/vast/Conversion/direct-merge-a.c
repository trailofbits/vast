// RUN: %vast-front -vast-emit-mlir-after=vast-emit-abi %s -o -  | %file-check %s -check-prefix=ABI

struct vec
{
    int a;
    int b;
};

// ABI:      abi.func {{.*}} (%arg0: i64) -> si32 {{.*}}
// ABI-NEXT: [[P0:%[0-9]+]] = abi.prologue {
// ABI-NEXT:   [[P9:%[0-9]+]] = abi.direct %arg0 : i64 -> !hl.record<@vec>
// ABI-NEXT:   {{.*}} = abi.yield [[P9]] : !hl.record<@vec> -> !hl.record<@vec>
// ABI-NEXT: } : !hl.record<@vec>

// ABI:      [[E8:%[0-9]+]] = abi.epilogue {
// ABI-NEXT:   [[E9:%[0-9]+]] = abi.direct {{.*}} : si32 -> si32
// ABI-NEXT:   {{.*}} = abi.yield [[E9]] : si32 -> si32
// ABI-NEXT: } : si32
// ABI-NEXT: ll.return [[E8]] : si32
int sum( struct vec v )
{
    return v.a + v.b;
}

// ABI:      [[C3:%[0-9]+]] = abi.call_exec @sum([[C2:%[0-9]+]]) {
// ABI-NEXT:   [[C4:%[0-9]+]] = abi.call_args {
// ABI-NEXT:     [[C8:%[0-9]+]] = abi.direct {{.*}} : !hl.record<@vec> -> i64
// ABI-NEXT:     {{.*}} = abi.yield [[C8]] : i64 -> i64
// ABI-NEXT:   } : i64
// ABI-NEXT:   [[C5:%[0-9]+]] = abi.call @sum([[C4]]) : (i64) -> si32
// ABI-NEXT:   [[C6:%[0-9]+]] = abi.call_rets {
// ABI-NEXT:     [[C8:%[0-9]+]] = abi.direct [[C5]] : si32 -> si32
// ABI-NEXT:     {{.*}} = abi.yield [[C8]] : si32 -> si32
// ABI-NEXT:   } : si32
// ABI-NEXT:   {{.*}} = abi.yield [[C6]] : si32 -> si32
// ABI-NEXT: } : (!hl.record<@vec>) -> si32
// ABI-NEXT: ll.return [[C3]] : si32
int main()
{
    struct vec v = { 0, 1 };
    return sum( v );
}
