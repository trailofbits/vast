set(VAST_OPTIONAL_SOURCES
  SymbolInterface.cpp
  TypeQualifiersInterfaces.cpp
  AliasTypeInterface.cpp
  DefaultDataLayoutTypeInterface.cpp
  ElementTypeInterface.cpp
  AggregateTypeDefinitionInterface.cpp
  ASTContextInterface.cpp
  ASTDeclInterface.cpp
  ASTStmtVisitorInterface.cpp
  ASTTypeInterface.cpp
  ASTExprInterface.cpp
)

add_vast_interface_library(AggregateTypeDefinitionInterface
  AggregateTypeDefinitionInterface.cpp
)

add_vast_interface_library(SymbolInterface
  SymbolInterface.cpp
)

add_vast_interface_library(TypeQualifiersInterfaces
  TypeQualifiersInterfaces.cpp
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

add_subdirectory(ClangAST)
