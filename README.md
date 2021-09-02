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

If you want to build with tests:

```
cmake \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX=<install directory> \
    -DVCPKG_ROOT=<vcpkg root> \
    -DVCPKG_TARGET_TRIPLET=<vcpkg triplet> \
    -ENABLE_TESTING=ON \
    -LLVM_EXTERNAL_LIT=<path to lit> \
    ..
```

## Run

To run mlir codegen of highlevel dialect use:

```
./build/bin/vast-cc --from-source <input.c>
```

## Test

```
cmake --build <build-dir> --target check-vast
```
