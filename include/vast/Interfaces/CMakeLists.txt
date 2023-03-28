add_mlir_interface(SymbolInterface)

set(LLVM_TARGET_DEFINITIONS TypeQualifiersInterfaces.td)
mlir_tablegen(TypeQualifiersInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TypeQualifiersInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRTypeQualifiersInterfacesIncGen)
