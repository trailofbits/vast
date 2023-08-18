# Copyright (c) 2023-present, Trail of Bits, Inc.

get_property(VAST_CONVERSION_LIBS GLOBAL PROPERTY VAST_CONVERSION_LIBS)

add_vast_library(CodeGen
    ABIInfo.cpp
    ArgInfo.cpp
    CodeGenScope.cpp
    CodeGenStmtVisitor.cpp
    CodeGenTypeVisitor.cpp
    CodeGen.cpp
    CodeGenDriver.cpp
    CodeGenTypeDriver.cpp
    CodeGenFunction.cpp
    DataLayout.cpp
    FunctionInfo.cpp
    Generator.cpp
    ItaniumCXXABI.cpp
    Mangler.cpp
    Passes.cpp
    TargetInfo.cpp
    TypeInfo.cpp

  LINK_LIBS PUBLIC
    ${CLANG_LIBS}
    ${VAST_CONVERSION_LIBS}
)
