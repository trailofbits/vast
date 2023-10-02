set(VAST_OPTIONAL_SOURCES
  SymbolInterface.cpp
  TypeQualifiersInterfaces.cpp
  DefaultDataLayoutTypeInterface.cpp
  ElementTypeInterface.cpp
)

add_vast_interface_library(SymbolInterface
  SymbolInterface.cpp
)

add_vast_interface_library(TypeQualifiersInterfaces
  TypeQualifiersInterfaces.cpp
)

add_vast_interface_library(DefaultDataLayoutTypeInterface
  DefaultDataLayoutTypeInterface.cpp
)

add_vast_interface_library(ElementTypeInterface
  ElementTypeInterface.cpp
)
