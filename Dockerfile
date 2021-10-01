ARG LLVM_VERSION=12.0.1
ARG UBUNTU_VERSION=21.04
ARG BUILD_BASE=ubuntu:${UBUNTU_VERSION}
ARG LIBRARIES=/opt/trailofbits

# Build-time dependencies go here
FROM ${BUILD_BASE} as deps
ARG LLVM_VERSION
ARG LIBRARIES

# Build dependencies
WORKDIR /dependencies

RUN apt-get update && \
    apt-get install -y software-properties-common wget tar

RUN wget "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-16.04.tar.xz" || \
    { echo 'Error downloading LLVM version ${LLVM_VERSION}' ; exit 1; }
RUN mkdir llvm-${LLVM_VERSION}
RUN tar -xf clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-16.04.tar.xz -C ./llvm-${LLVM_VERSION} --strip-components=1 && \
    rm clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-16.04.tar.xz

RUN apt-get update && \
    apt-get install -y clang-12 libstdc++-11-dev cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Source code build
FROM deps AS build
WORKDIR /vast
ARG LLVM_VERSION
ARG LIBRARIES

COPY . ./

## Build vast
RUN cmake -G Ninja -B build -S . \
    -DCMAKE_C_COMPILER=clang-12 \
    -DCMAKE_CXX_COMPILER=clang++-12 \
    -DCMAKE_INSTALL_PREFIX="${LIBRARIES}" \
    -DCMAKE_VERBOSE_MAKEFILE=True \
    -DLLVM_INSTALL_DIR=/dependencies/llvm-${LLVM_VERSION}
RUN cmake --build build --target install
