# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11, Ubuntu 18.04/22.04 on local arm64/Apple Silicon): debian-11, debian-10, ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG IMAGE="ubuntu-22.04"
ARG LLVM_VERSION="17"

FROM mcr.microsoft.com/vscode/devcontainers/cpp:${IMAGE}
ARG LLVM_VERSION

# [Optional] Install CMake version different from what base image has already installed.
# CMake reinstall choices: none, 3.21.5, 3.22.2, or versions from https://cmake.org/download/
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.29.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

COPY ./dependencies.txt /tmp/
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    xargs -a /tmp/dependencies.txt apt-get -y install --no-install-recommends && \
    apt-get clean --yes && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get -y purge llvm-14 clang-14

RUN pip3 install lit

COPY ./install-llvm.sh /tmp/
RUN chmod +x /tmp/install-llvm.sh && /tmp/install-llvm.sh ${LLVM_VERSION} all
