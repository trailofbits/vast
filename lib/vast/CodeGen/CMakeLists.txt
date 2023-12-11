# Copyright (c) 2023-present, Trail of Bits, Inc.

get_property(VAST_CONVERSION_LIBS GLOBAL PROPERTY VAST_CONVERSION_LIBS)

add_vast_library(CodeGen
    CodeGenScope.cpp
    CodeGenStmtVisitor.cpp
    CodeGenTypeVisitor.cpp
    CodeGen.cpp
    CodeGenDriver.cpp
    CodeGenFunction.cpp
    DataLayout.cpp
    Mangler.cpp

  LINK_LIBS PUBLIC
    ${CLANG_LIBS}
    ${VAST_CONVERSION_LIBS}
)
