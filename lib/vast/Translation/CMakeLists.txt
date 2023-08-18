# FIXME: split to two libraries folders
set(VAST_OPTIONAL_SOURCES
  CodeGenScope.cpp
  CodeGenStmtVisitor.cpp
  CodeGenTypeVisitor.cpp
  DataLayout.cpp
  CodeGen.cpp
  CodeGenDriver.cpp
  CodeGenTypeDriver.cpp
  CodeGenFunction.cpp
  Mangler.cpp

  FromSource.cpp
)

add_vast_library( Translation
  CodeGenScope.cpp
  CodeGenStmtVisitor.cpp
  CodeGenTypeVisitor.cpp
  DataLayout.cpp
  CodeGen.cpp
  CodeGenDriver.cpp
  CodeGenTypeDriver.cpp
  CodeGenFunction.cpp
  Mangler.cpp
)

add_vast_library( FromSourceParser
  FromSource.cpp
)
