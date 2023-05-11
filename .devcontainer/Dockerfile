# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11, Ubuntu 18.04/22.04 on local arm64/Apple Silicon): debian-11, debian-10, ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="ubuntu-22.04"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# [Optional] Install CMake version different from what base image has already installed.
# CMake reinstall choices: none, 3.21.5, 3.22.2, or versions from https://cmake.org/download/
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.23.3"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        software-properties-common python3-pip ccache cmake lld

RUN pip3 install lit

# Install LLVM 16
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
RUN add-apt-repository 'deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main'

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        clang-16 clang-tools-16 libclang-common-16-dev libclang-16-dev \
        libllvm16 libmlir-16-dev mlir-16-tools llvm-16 llvm-16-dev libclang-rt-16-dev

RUN sudo ln -s /usr/bin/FileCheck-16 /usr/bin/FileCheck

RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-16 60
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-16 60
