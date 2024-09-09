set(VAST_OPTIONAL_SOURCES
  AliasTypeInterface.cpp
  DefaultDataLayoutTypeInterface.cpp
  ElementTypeInterface.cpp
  SymbolInterface.cpp
  SymbolTableInterface.cpp
  SymbolRefInterface.cpp
  TypeQualifiersInterfaces.cpp
  TypeTraitExprInterface.cpp
)

add_vast_interface_library(AliasTypeInterface
  AliasTypeInterface.cpp
)

add_vast_interface_library(DefaultDataLayoutTypeInterface
  DefaultDataLayoutTypeInterface.cpp
)

add_vast_interface_library(ElementTypeInterface
  ElementTypeInterface.cpp
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

add_vast_interface_library(TypeQualifiersInterfaces
  TypeQualifiersInterfaces.cpp
)

add_vast_interface_library(TypeTraitExprInterface
  TypeTraitExprInterface.cpp
)

add_subdirectory(AST)
