// RUN: %vast-front -vast-emit-mlir-after=vast-emit-abi %s -o -  | %file-check %s -check-prefix=ABI

struct wrapped
{
    int x;
};

int fn( struct wrapped w )
{
    return w.x;
}

int main()
{
    // ABI:      {{.*}} = abi.call_exec @fn({{.*}}) {
    // ABI-NEXT:   [[V4:%[0-9]+]] = abi.call_args {
    // ABI-NEXT:     [[V8:%[0-9]+]] = abi.direct {{.*}} : !hl.record<@wrapped> -> i32
    // ABI-NEXT:     {{.*}} = abi.yield [[V8]] : i32 -> i32
    // ABI-NEXT:   } : i32
    // ABI-NEXT:   [[V5:%[0-9]+]] = abi.call @fn([[V4]]) : (i32) -> si32
    // ABI-NEXT:   [[V6:%[0-9]+]] = abi.call_rets {
    // ABI-NEXT:     [[V9:%[0-9]+]] = abi.direct [[V5]] : si32 -> si32
    // ABI-NEXT:     {{.*}} = abi.yield [[V9]] : si32 -> si32
    // ABI-NEXT:   } : si32
    // ABI-NEXT:   {{.*}} = abi.yield [[V6]] : si32 -> si32
    // ABI-NEXT: } : (!hl.record<@wrapped>) -> si32
    struct wrapped w;
    fn( w );
}
