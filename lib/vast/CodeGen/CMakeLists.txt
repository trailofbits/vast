# Copyright (c) 2023-present, Trail of Bits, Inc.

get_property(VAST_CONVERSION_LIBS GLOBAL PROPERTY VAST_CONVERSION_LIBS)

add_vast_library(CodeGen
    CodeGenDriver.cpp

    CodeGenStmtVisitor.cpp
    CodeGenTypeVisitor.cpp
    CodeGenVisitor.cpp

    CodeGenFunction.cpp
    CodeGenModule.cpp

    DataLayout.cpp
    Mangler.cpp

    UnsupportedVisitor.cpp

  LINK_LIBS PUBLIC
    ${CLANG_LIBS}
    ${VAST_CONVERSION_LIBS}
)
