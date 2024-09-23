set(VAST_OPTIONAL_SOURCES
  DeclStorageInterface.cpp
  SymbolInterface.cpp
  SymbolTableInterface.cpp
  SymbolRefInterface.cpp
  TypeDefinitionInterface.cpp
)

add_vast_interface_library(DeclStorageInterface
    DeclStorageInterface.cpp
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
