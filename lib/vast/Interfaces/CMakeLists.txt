set(LLVM_OPTIONAL_SOURCES
  SymbolInterface.cpp
  TypeQualifiersInterfaces.cpp
)

function(add_vast_interface_library name)
  add_mlir_library(VAST${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast/Interfaces

    DEPENDS
    MLIR${name}IncGen

    LINK_LIBS PUBLIC
    MLIRIR
  )
endfunction(add_vast_interface_library)

add_vast_interface_library(SymbolInterface)
add_vast_interface_library(TypeQualifiersInterfaces)
