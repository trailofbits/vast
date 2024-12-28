# Progress Sheet

Skipping any duplicate libc calls sequentially down the file(s)

## `basename-lgpl.o.mlir`

**libc functions imported**
`hl.func @strlen external (!hl.lvalue<!hl.ptr<!hl.char< const >>>) -> !hl.long< unsigned > attributes {hl.builtin = #hl.builtin<1148>, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc2)` (5)

**Non-transformed libc calls**
`%3 = hl.call @strlen(%2) : (!hl.ptr<!hl.char< const >>) -> !hl.long< unsigned > loc(#loc41)` (107)

**Non-transformed arithmetic operations**
`%10 = hl.add %8, %9 : (!hl.ptr<!hl.char< const >>, !hl.int) -> !hl.ptr<!hl.char< const >> loc(#loc8)` (12)

**Local Functions**
`@last_component` (6)
`@base_len` (96)

## `bits.o.mlir`

**libc-esque functions imported**
`hl.func @__builtin_bswap32 external (!hl.lvalue<!hl.int< unsigned >>) -> !hl.int< unsigned > attributes {hl.builtin = #hl.builtin<180>, hl.const = #hl.const, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc2)` (15)
`hl.func @__builtin_bswap64 external (!hl.lvalue<!hl.longlong< unsigned >>) -> !hl.longlong< unsigned > attributes {hl.builtin = #hl.builtin<181>, hl.const = #hl.const, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc3)` (16)

**Local Functions**
`@flush_outbuf`
`@bi_windup`

## `calloc.o.mlir`

Same issue. Should verify.

**libc functions imported**
`hl.func @calloc external (!hl.lvalue<!hl.long< unsigned >>, !hl.lvalue<!hl.long< unsigned >>) -> !hl.ptr<!hl.void> attributes {hl.alloc_size = #hl.alloc_size<size_pos : 1, num_pos : 2>, hl.builtin = #hl.builtin<256>, hl.warn_unused_result = #hl.warn_unused_result, sym_visibility = "private"} loc(#loc31)` (52)
`hl.func @__error external () -> !hl.ptr<!hl.int> attributes {sym_visibility = "private"} loc(#loc32)` -> errno is technically libc, but __error() looks like kernel

**Non-transformed libc calls**
`%5 = hl.call @calloc(%2, %4) : (!hl.long< unsigned >, !hl.long< unsigned >) -> !hl.ptr<!hl.void> loc(#loc67)` (124)

## `chdir-long.o.mlir`

**libc functions**

`hl.func @chdir external (!hl.lvalue<!hl.ptr<!hl.char< const >>>) -> !hl.int attributes {sym_visibility = "private"} loc(#loc5)` (17)
`hl.func @fchdir external (!hl.lvalue<!hl.int>) -> !hl.int attributes {sym_visibility = "private"} loc(#loc7)` (19)
`hl.func @__assert_rtn external (!hl.lvalue<!hl.ptr<!hl.char< const >>>, !hl.lvalue<!hl.ptr<!hl.char< const >>>, !hl.lvalue<!hl.int>, !hl.lvalue<!hl.ptr<!hl.char< const >>>) -> !hl.void attributes {hl.cold = #hl.cold, hl.disable_tail_calls = #hl.disable_tail_calls, sym_visibility = "private"} loc(#loc73)` (136)
`%6 = hl.call @"\01_close"(%5) : (!hl.int) -> !hl.int loc(#loc99)` (179) --> NOT libc (internal function), but replaced close anyways.
`// #loc99 = loc("/Users/shinlee/tob/gzip/src/gzip-1.10/lib/chdir-long.c":63:25)`
`hl.func @__builtin_expect external (!hl.lvalue<!hl.long>, !hl.lvalue<!hl.long>) -> !hl.long attributes {hl.builtin = #hl.builtin<528>, hl.const = #hl.const, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc4)` (16)
IMPORTANT: Location is "unknown", because it's a gcc function. Replacing for now.
`hl.func @"\01_close" external (!hl.lvalue<!hl.int>) -> !hl.int attributes {hl.asm = #hl.asm<"_close", true>, sym_visibility = "private"} loc(#loc6)` (18)
`hl.func @rpl_openat external (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.ptr<!hl.char< const >>>, !hl.lvalue<!hl.int>, ...) -> !hl.int attributes {hl.nonnull = #hl.nonnull, sym_visibility = "private"} loc(#loc4)` (131)
`hl.func @cdb_free internal (%arg0: !hl.lvalue<!hl.ptr<!hl.record<@cd_buf>>> loc("/Users/shinlee/tob/gzip/src/gzip-1.10/lib/chdir-long.c":59:32)) -> !hl.void attributes {sym_visibility = "private"} {` (162)
`hl.func @strspn external (!hl.lvalue<!hl.ptr<!hl.char< const >>>, !hl.lvalue<!hl.ptr<!hl.char< const >>>) -> !hl.long< unsigned > attributes {hl.builtin = #hl.builtin<1156>, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc72)` (134)
`hl.func @cdb_init internal (%arg0: !hl.lvalue<!hl.ptr<!hl.record<@cd_buf>>> loc("/Users/shinlee/tob/gzip/src/gzip-1.10/lib/chdir-long.c":47:26)) -> !hl.void attributes {sym_visibility = "private"} {` (140)
`hl.func @memchr external (!hl.lvalue<!hl.ptr<!hl.void< const >>>, !hl.lvalue<!hl.int>, !hl.lvalue<!hl.long< unsigned >>) -> !hl.ptr<!hl.void> attributes {hl.builtin = #hl.builtin<863>, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc70)` (132)
`hl.func @cdb_advance_fd internal (%arg0: !hl.lvalue<!hl.ptr<!hl.record<@cd_buf>>> loc("/Users/shinlee/tob/gzip/src/gzip-1.10/lib/chdir-long.c":73:32), %arg1: !hl.lvalue<!hl.ptr<!hl.char< const >>> loc("/Users/shinlee/tob/gzip/src/gzip-1.10/lib/chdir-long.c":73:49)) -> !hl.int attributes {sym_visibility = "private"} {` (222)
---

## `cloexec.o.mlir`

**libc functions**
`hl.func @"\01_fcntl" external (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.int>, ...) -> !hl.int attributes {hl.asm = #hl.asm<"_fcntl", true>, sym_visibility = "private"} loc(#loc61)` (116) - verify name

## `creat-safer.o.mlir`

**libc functions**
`hl.func @"\01_creat" external (!hl.lvalue<!hl.ptr<!hl.char< const >>>, !hl.lvalue<!hl.short< unsigned >>) -> !hl.int attributes {hl.asm = #hl.asm<"_creat", true>, sym_visibility = "private"} loc(#loc61)` (115)

## `deflate.o.mlir`

**libc functions**
`hl.func @__builtin_object_size external (!hl.lvalue<!hl.ptr<!hl.void< const >>>, !hl.lvalue<!hl.int>) -> !hl.long< unsigned > attributes {hl.builtin = #hl.builtin<957>, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc4)` (15)
`hl.func @__builtin___memset_chk external (!hl.lvalue<!hl.ptr<!hl.void>>, !hl.lvalue<!hl.int>, !hl.lvalue<!hl.long< unsigned >>, !hl.lvalue<!hl.long< unsigned >>) -> !hl.ptr<!hl.void> attributes {hl.builtin = #hl.builtin<877>, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc4)`
`hl.func @__builtin___memcpy_chk external (!hl.lvalue<!hl.ptr<!hl.void>>, !hl.lvalue<!hl.ptr<!hl.void< const >>>, !hl.lvalue<!hl.long< unsigned >>, !hl.lvalue<!hl.long< unsigned >>) -> !hl.ptr<!hl.void> attributes {hl.builtin = #hl.builtin<873>, hl.nothrow = #hl.nothrow, sym_visibility = "private"} loc(#loc4)` (16)

## `dirname-lgpl.o.mlir`
(Redundant functions only)

## `dup-safer-flag.o.mlir`
(Redundant functions only)

## `dup-safer.o.mlir`
(Redundant functions only)

## `error.o.mlir`

**libc functions**
`hl.func @fileno external (!hl.lvalue<!hl.ptr<!hl.record<@__sFILE>>>) -> !hl.int attributes {sym_visibility = "private"} loc(#loc35)`
`hl.func @rpl_fflush external (!hl.lvalue<!hl.ptr<!hl.record<@__sFILE>>>) -> !hl.int attributes {sym_visibility = "private"} loc(#loc59)`
`hl.func @rpl_fprintf external (!hl.lvalue<!hl.ptr<!hl.record<@__sFILE>>>, !hl.lvalue<!hl.ptr<!hl.char< const >>>, ...) -> !hl.int attributes {hl.format = #hl.format<"printf">, hl.nonnull = #hl.nonnull, sym_visibility = "private"} loc(#loc59)` (96)
`hl.func @rpl_vfprintf external (!hl.lvalue<!hl.ptr<!hl.record<@__sFILE>>>, !hl.lvalue<!hl.ptr<!hl.char< const >>>, !hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.int attributes {hl.format = #hl.format<"printf">, hl.nonnull = #hl.nonnull, sym_visibility = "private"} loc(#loc59)` (97)
`putc` (56)
`getprogname` (104)
---

## LibC transformations

- `strlen` as `sink`
- `calloc` as `nonparser`
- `__builtin_bswap32` as `sink` 
- `__builtin_bswap64` as `sink`
- `chdir` as `nonparser`
- `fchdir` as `nonparser`
- `close` as `nonparser` TODO: name is "\01_close", which prob refers to internal function to distinguish. 
- `__builtin_expect` as `nonparser`
- `__assert_rtn` as `nonparser`
- `rpl_openat` as `sink`
- `cdb_free` as `nonparser`
- `cdb_init` as 
- `strspn` as `parser`
- `__error` as `nonparser`
<!-- - `cdb_init` as `nonparser` TODO: Cannot find definition with only one param.-->
-  `memchr` as `parser`
<!-- - `cdb_advance_fd` as TODO: Cannot find definition :( -->
- `fcntl` as `nonparser`
- `creat` as `sink`
- `__builtin_object_size` as `nonparser` TODO: Verify
- `__builtin___memset_chk` as `nonparser`
- `__builtin___memcpy_chk` as `nonparser`
- `fileno` as `sink` (maybe source, double check)
- `rpl_fflush` as `sink` (since 'flushing' data)
- `rpl_strerror_r` as `nonparser`
- `rpl_fprintf ` as `sink`
- `rpl_vfprintf` as `sink`
- `putc` as `sink`
- `exit` as `nonparser` (already in model, no action necessary)
- `getprogname` as `data` TODO! verify