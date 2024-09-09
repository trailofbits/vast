set(VAST_OPTIONAL_SOURCES
  SymbolInterface.cpp
  SymbolTableInterface.cpp
  SymbolRefInterface.cpp
  TypeDefinitionInterface.cpp
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
