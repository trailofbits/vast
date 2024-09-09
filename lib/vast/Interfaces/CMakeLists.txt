set(VAST_OPTIONAL_SOURCES
  AliasTypeInterface.cpp
  DefaultDataLayoutTypeInterface.cpp
  ElementTypeInterface.cpp
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

add_vast_interface_library(TypeQualifiersInterfaces
  TypeQualifiersInterfaces.cpp
)

add_vast_interface_library(TypeTraitExprInterface
  TypeTraitExprInterface.cpp
)

add_subdirectory(AST)
