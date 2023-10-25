// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-emit-abi | %file-check %s -check-prefix=ABI

struct data
{
    int a;
    short b;
    unsigned long c;
};

// ABI:      abi.func {{.*}} (%arg0: !hl.lvalue<i64>) -> si32 {{.*}}
// ABI:        {{.*}} = abi.prologue {
// ABI-NEXT:     [[V0:%[0-9]+]] = abi.direct %arg0 : !hl.lvalue<i64> -> !hl.lvalue<!hl.elaborated<!hl.record<"data">>>
// ABI-NEXT:     {{.*}} = abi.yield [[V0]] : !hl.lvalue<!hl.elaborated<!hl.record<"data">>> -> si32
// ABI-NEXT:   } : !hl.lvalue<!hl.elaborated<!hl.record<"data">>>


// ABI:       [[V5:%[0-9]+]] = abi.epilogue {
// ABI-NEXT:      [[V6:%[0-9]+]] = abi.direct {{.*}} : si32 -> si32
// ABI-NEXT:      {{.*}} = abi.yield [[V6]] : si32 -> si32
// ABI-NEXT:    } : si32
// ABI-NEXT:    hl.return [[V5]] : si32
// ABI-NEXT:  }
int sum( struct data d )
{
    return d.a + d.b + d.c;
}

// ABI:      [[V13:%[0-9]+]] = abi.call_exec @sum([[V12:%[0-9]+]]) {
// ABI-NEXT:   [[V14:%[0-9]+]] = abi.call_args {
// ABI-NEXT:     [[V18:%[0-9]+]] = abi.direct [[V12]] : !hl.elaborated<!hl.record<"data">> -> i64
// ABI-NEXT:     {{.*}} = abi.yield [[V18]] : i64 -> i64
// ABI-NEXT:   } : i64
// ABI-NEXT:   [[V15:%[0-9]+]] = abi.call @sum([[V14]]) : (i64) -> si32
// ABI-NEXT:   [[V16:%[0-9]+]] = abi.call_rets {
// ABI-NEXT:     [[V28:%[0-9]+]] = abi.direct [[V15]] : si32 -> si32
// ABI-NEXT:     {{.*}} = abi.yield [[V28]] : si32 -> si32
// ABI-NEXT:   } : si32
// ABI-NEXT:   %7 = abi.yield [[V16]] : si32 -> si32
// ABI-NEXT: } : (!hl.elaborated<!hl.record<"data">>) -> si32
// ABI-NEXT: hl.return [[V13]] : si32

int main()
{
    struct data d = { 0, 1, 2 };
    return sum( d );
}
