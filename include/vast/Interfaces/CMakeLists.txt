add_mlir_interface(SymbolInterface)

set(LLVM_TARGET_DEFINITIONS TypedAttrInterface.td)
mlir_tablegen(TypedAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(TypedAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRTypedAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS TypeQualifiersInterfaces.td)
mlir_tablegen(TypeQualifiersInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeQualifiersInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRTypeQualifiersInterfacesIncGen)
