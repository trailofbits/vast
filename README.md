# VAST — Verbose AST


## Build

```
cmake \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX=<install directory> \
    -DVCPKG_ROOT=<vcpkg root> \
    -DVCPKG_TARGET_TRIPLET=<vcpkg triplet> \
    ..
```

## Run

To run mlir codegen of highlevel dialect use:

```
./build/bin/vast-cc --from-source <input.c>
```
