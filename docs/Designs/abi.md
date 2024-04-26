## ABI

VAST partially models ABI specifications for function types and therefore
callsites. While the specification goes into details regarding registers, for
now VAST only offers lowering similar to what clang codegen does - argument and
return types are coerced to types that will easily fit their respective
registers once that allocation takes place. There is nothing preventing
inclusion of information about registers as well (for example as metadata or
separate operations/dialect), however it is not yet implemented.

Similar to other transformation in VAST, ABI modelling tries to be as modular as
possible and as such can be split into three distinct steps:

- Compute classification of types
- Encode computed classification into module
- Lower transformed module into some "executable" dialect

Main goal for now is to lower function prototypes to match the types produced by
clang, so that VAST emitted LLVM can be used as drop-in replacement for clang
one.

When reading this document please keep in mind that implementation of this
feature is still ongoing and therefore particular technical details could change
drastically (although we hope that overall design will remain the same).

## Classification

Mirrors what clang does, but instead of locking the computed information away,
we expose the API. In ideal world we would like to keep the algorithm(s, as
there may be multiple per different ABIs) generic. This can be achieved by
requiring users to implement & provide interface that specifies various details
about used types; algorithm will be same when talking about `hl` or `LLVM` types
after all.

We currently implement classification algorithm for x86 as it is our main target
and it provides nice test of the approach given all of the weird cases that
can happen.

## ABI Dialect (`-vast-emit-abi`)

Once classification for a function is computed, we need to:

- Change function prototype
- Encode how new types match to the old types + some oddities such as `sret`.

To facilitate this, VAST contains `abi` dialect, which operations encode
"high-level" descriptions of type transformations that can occur during ABI
lowering as operations. This is not very different from what clang does, but
VAST does it over multiple steps.

For functions, type change itself is easy and to mark that function is
transformed, `abi.func` operation is used instead of original one to define the
newly formed function. However, as arguments and return types are different, we
introduce `abi.prologue` and `abi.epilogue` operations.

Consider following function we want to lower:

Disclaimer: Since `abi` dialect does not have nice formatting, therefore examples in
this section contain some artistic liberty, but semantics (and operations) are
preserved.


```
strut Point{ int; int; int; };

int size( struct Point p ) { ... }
```

After running the classification, we discover that type of `size` should be
`( i64, i32 ) -> i32` - both arguments and
returned value are passed directly. Therefore we encode it as follows:

```
abi.func size(i64 %arg0_0, i32 %arg0_1 ) -> i32
{
    %arg = abi.prologue {
      %4 = abi.direct %arg0, %arg1 : i64, i32 -> !hl.record<"Point">
      %5 = abi.yield %4 : !hl.record<"Point"> -> !hl.record<"Point">
    } : !hl.record<"Point">

    // Computation can continue as before, because %arg has a correct type

    %ret = ... value that was previously returned ... -> i32
    %out = abi.epilogue
    {
        %0 = abi.direct %ret: i32 -> i32
        abi.yield %0
    }
    hl.return %out
}
```

In case, there were multiple function arguments, the `abi.prologue` would return
more values.

```

%args = abi.prologue -> (hl.struct< "Point" >, i32 )
{
    %0 = abi.direct %arg0_0, %arg0_1
    %1 = abi.direct %arg1
    abi.yield %0, %1
}
```


This design allows easy analysis and subsequent rewrite (as each function has a
prologue and epilogue and returned values are explicitly yielded).

Callsites are transformed in the same manner (unfortunately, they look more
complicated due to nested regions):

```

%x = hl.call< "size" > %arg: hl.struct< "Point" > -> i32

%x = abi.call< "size" >: () -> i32
{
    %arg0_0, %arg0_1 = abi.call_args: () -> (i64, i32)
    {
        %0, %1 = abi.direct %arg
        abi.yield %0, %1
    }
    %x' = hl.call< "size" > %arg0_0, &arg0_1 : (i64, i32) -> i32
    %0 = abi.call_rets : () -> i32
    {
        %0 = abi.direct %x' : i32 -> i32
        abi.yield %0
    }
    abi.yield %0
}

```

If we had an argument passed as `MEMORY` class, we would encode it in a similar manner.
```
  ll.func @foo external (%arg0: !hl.record<"data">) -> si32 {
```
Gets transformed to:
```
  abi.func @vast.abi.foo external (%arg0: !hl.ptr<!hl.record<"data">>) -> si32 {
    %0 = abi.prologue {
      %5 = abi.indirect %arg0 : !hl.ptr<!hl.record<"data">> -> !hl.record<"data">
      %6 = abi.yield %5 : !hl.record<"data"> -> !hl.record<"data">
    } : !hl.record<"data">
```
Call site:
```
    %2 = abi.call_exec @da(%1) {
      %3 = abi.call_args {
        %7 = abi.indirect %1 : !hl.record<"data"> -> !hl.ptr<!hl.record<"data">>
        %8 = abi.yield %7 : !hl.ptr<!hl.record<"data">> -> !hl.ptr<!hl.record<"data">>
      } : !hl.ptr<!hl.record<"data">>
      %4 = abi.call @da(%3) : (!hl.ptr<!hl.record<"data">>) -> si32
      %5 = abi.call_rets {
        %7 = abi.direct %4 : si32 -> si32
        %8 = abi.yield %7 : si32 -> si32
      } : si32
      %6 = abi.yield %5 : si32 -> si32
    } : (!hl.record<"data">) -> si32
```

For now, same `abi` operations are used to encode transformation in callsite and
function (although they change the value in a "opposite direction"), this may be
later revisited, but for now it is enough to look at the parent operation to
determine whether the transformation lies in a function or callsite.

## Lowering to some executable dialect (`-vast-lower-abi`)

While `abi` dialect provides us with all the information required to do the
transformation, it does not "compute" anything. Rather this lowering is left to
a next pass. We hope by splitting the transformation into 2,
we achieve the following:

- We can implement multiple "backends" - whether back to `hl`, `llvm` or totally
  random dialect of user choice.
- Re-use existing implementation of classification algorithm.

Currently we lower into our own dialect stack. To continue with our example -
after lowering the prologue:

```
  ll.func @size external (%arg0: i64, %arg1: i32) -> si32 {
    %0 = ll.extract %arg0 {from = 0 : ui64, to = 32 : ui64} : (i64) -> si32
    %1 = ll.extract %arg0 {from = 32 : ui64, to = 64 : ui64} : (i64) -> si32
    %2 = hl.initlist %0, %1, %arg1 : (si32, si32, i32) -> !hl.record<"Point">
    %3 = ll.alloca : !hl.ptr<!hl.record<"Point">>
    ll.store %3, %2 : !hl.ptr<!hl.record<"Point">>, !hl.record<"Point">
```
And the callsite:
```
    %9 = "ll.gep"(%8) <{idx = 0 : i32, name = "a"}> : (!hl.ptr<!hl.record<"Point">>) -> !hl.ptr<si32>
    %10 = ll.load %9 : (!hl.ptr<si32>) -> si32
    %11 = "ll.gep"(%8) <{idx = 1 : i32, name = "b"}> : (!hl.ptr<!hl.record<"Point">>) -> !hl.ptr<si32>
    %12 = ll.load %11 : (!hl.ptr<si32>) -> si32
    %13 = ll.concat %10, %12 : (si32, si32) -> i64
    %14 = "ll.gep"(%8) <{idx = 2 : i32, name = "c"}> : (!hl.ptr<!hl.record<"Point">>) -> !hl.ptr<si32>
    %15 = ll.load %14 : (!hl.ptr<si32>) -> si32
    %16 = ll.concat %15 : (si32) -> i32
    %17 = hl.call @size(%13, %16) : (i64, i32) -> si32
    ll.store %5, %17 : !hl.ptr<si32>, si32
    %18 = hl.const #core.integer<0> : si32
```

We do not use `memcpy` as we try to preserve as much explicit data flow as we
can but nothing really prevents it.
