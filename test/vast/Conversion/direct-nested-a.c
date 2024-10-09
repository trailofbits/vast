// RUN: %vast-front -vast-emit-mlir-after=vast-emit-abi %s -o -  | %file-check %s -check-prefix=ABI

struct pad_24
{
    char a;
    short b;
};

struct data
{
    int a;
    short b;
    struct pad_24 pad;
};

// ABI:      abi.func {{.*}} (%arg0: i64) -> i64 {{.*}}
// ABI:        {{.*}} = abi.prologue {
// ABI-NEXT:     [[P4:%[0-9]+]] = abi.direct %arg0 : i64 -> !hl.record<@data>
// ABI-NEXT:     {{.*}} = abi.yield [[P4]] : !hl.record<@data> -> !hl.record<@data>
// ABI-NEXT:   } : !hl.record<@data>

// ABI:      [[E3:%[0-9]+]] = abi.epilogue {
// ABI-NEXT:   [[E4:%[0-9]+]] = abi.direct {{.*}} : !hl.record<@data> -> i64
// ABI-NEXT:   {{.*}} = abi.yield [[E4]] : i64 -> i64
// ABI-NEXT: } : i64
// ABI-NEXT: ll.return [[E3]] : i64

struct data id( struct data v )
{
    return v;
}

// ABI:        {{.*}} = abi.call_exec @id([[C7:%[0-9]+]]) {
// ABI-NEXT:     [[C9:%[0-9]+]] = abi.call_args {
// ABI-NEXT:       [[C13:%[0-9]+]] = abi.direct [[C7]] : !hl.record<@data> -> i64
// ABI-NEXT:       {{.*}} = abi.yield [[C13]] : i64 -> i64
// ABI-NEXT:     } : i64
// ABI-NEXT:     [[C10:%[0-9]+]] = abi.call @id([[C9]]) : (i64) -> i64
// ABI-NEXT:     [[C11:%[0-9]+]] = abi.call_rets {
// ABI-NEXT:       [[C13:%[0-9]+]] = abi.direct [[C10]] : i64 -> !hl.record<@data>
// ABI-NEXT:       {{.*}} = abi.yield [[C13]] : !hl.record<@data> -> !hl.record<@data>
// ABI-NEXT:     } : !hl.record<@data>
// ABI-NEXT:     {{.*}} = abi.yield [[C11]] : !hl.record<@data> -> !hl.record<@data>
// ABI-NEXT:   } : (!hl.record<@data>) -> !hl.record<@data>
int main()
{
    struct pad_24 p = { 1, 2 };
    struct data v = { 0, 1, p };
    struct data x = id( v );
    return x.a;
}
