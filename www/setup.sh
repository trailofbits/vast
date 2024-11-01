#!/usr/bin/env bash

# Check if the current directory is a Git repository
if !( [ -d .git ] || git rev-parse --git-dir > /dev/null 2>&1 ); then
    echo "Error: Script must be run from the root of repository."
    exit 1
fi

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <destination> <docs_build_dir>"
    exit 1
fi

dst="$1"
build="$2"

# Check if the build directory exists
if [ ! -d "$build" ]; then
    echo "Error: Build directory '$build' not found."
    exit 1
fi

if [ ! -d "$build/docs" ]; then
    echo "Error: Build docs directory '$build/docs' not found."
    echo "Info: Probably you did not build `vast-doc` target."
    exit 1
fi

# Check if the destination directory exists, create it if not
if [ ! -d "$dst" ]; then
    echo "Creating destination directory: $dst"
    mkdir -p "$dst"
fi


# Setup hand-written docs
cp -rv $(pwd)/docs $dst
cp -rv $(pwd)/LICENSE $dst/docs
cp -rv $(pwd)/CONTRIBUTING.md $dst/docs

# Setup auto-generated docs
cp -rv $build/docs $dst/docs/dialects

# Setup benchmark results
if [ ! -d "$dst/docs/Benchmarks" ]; then
    echo "Creating benchmarks directory: $dst/docs/Benchmarks"
    mkdir -p "$dst/docs/Benchmarks"
fi

cp -rv $(pwd)/llvm-test-suite-results/single-source-results.md $dst/docs/Benchmarks/
cp -rv $(pwd)/sv-comp-results/sv-comp-results.md $dst/docs/Benchmarks/
cp -rv $(pwd)/linux_kernel_times/linux_kernel_times.md $dst/docs/Benchmarks/
cp -rv $(pwd)/gzip_times/gzip_times.md $dst/docs/Benchmarks/
cp -rv $(pwd)/gzip_times_1.10/gzip_times_1.10.md $dst/docs/Benchmarks/
cp -rv $(pwd)/gzip_times_1.13/gzip_times_1.13.md $dst/docs/Benchmarks/

# Setup site assets
cp -rv $(pwd)/www/assets $dst
cp -rv $(pwd)/www/mkdocs.yml $dst
