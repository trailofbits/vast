#!/bin/bash

# Set default LLVM version if not specified
LLVM_VERSION=${LLVM_VERSION:-19}

# Install LLVM tools and libraries
bash -c "$(curl -s -o - https://apt.llvm.org/llvm.sh)" llvm.sh $LLVM_VERSION

# Install required packages dynamically based on the LLVM version
apt-get update && apt-get install -y -q \
 libstdc++-12-dev \
 llvm-${LLVM_VERSION} \
 libmlir-${LLVM_VERSION} \
 libmlir-${LLVM_VERSION}-dev \
 mlir-${LLVM_VERSION}-tools \
 libclang-${LLVM_VERSION}-dev
