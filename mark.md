TODO: remove after checking replacements

legacy
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

