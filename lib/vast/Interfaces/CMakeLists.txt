set(VAST_OPTIONAL_SOURCES
  AggregateTypeDefinitionInterface.cpp
  AliasTypeInterface.cpp
  DefaultDataLayoutTypeInterface.cpp
  ElementTypeInterface.cpp
  SymbolInterface.cpp
  SymbolRefInterface.cpp
  TypeQualifiersInterfaces.cpp
)

add_vast_interface_library(AggregateTypeDefinitionInterface
  AggregateTypeDefinitionInterface.cpp
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

add_vast_interface_library(SymbolRefInterface
  SymbolRefInterface.cpp
)

add_vast_interface_library(TypeQualifiersInterfaces
  TypeQualifiersInterfaces.cpp
)

add_subdirectory(AST)
