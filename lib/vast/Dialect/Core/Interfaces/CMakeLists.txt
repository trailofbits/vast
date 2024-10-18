set(VAST_OPTIONAL_SOURCES
  DeclStorageInterface.cpp
  DesugarTypeInterface.cpp
  FunctionInterface.cpp
  FunctionImplementation.cpp
  SymbolInterface.cpp
  SymbolTableInterface.cpp
  SymbolRefInterface.cpp
  TypeDefinitionInterface.cpp
)

add_vast_interface_library(DeclStorageInterface
    DeclStorageInterface.cpp
)

add_vast_interface_library(DesugarTypeInterface
    DesugarTypeInterface.cpp
)

add_vast_interface_library(FunctionInterface
    FunctionInterface.cpp
    FunctionImplementation.cpp
)

add_vast_interface_library(SymbolInterface
  SymbolInterface.cpp
)

add_vast_interface_library(SymbolTableInterface
  SymbolTableInterface.cpp

  LINK_LIBS
    VASTSymbolInterface
)

add_vast_interface_library(SymbolRefInterface
  SymbolRefInterface.cpp
)

add_vast_interface_library(TypeDefinitionInterface
    TypeDefinitionInterface.cpp
)
