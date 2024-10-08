//===- FunctionSupport.h - Utility types for function-like ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support types for Operations that represent function-like
// constructs to use.
//
//===----------------------------------------------------------------------===//

#ifndef VAST_DIALECT_CORE_FUNCTIONINTERFACE_HPP
#define VAST_DIALECT_CORE_FUNCTIONINTERFACE_HPP

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallString.h>
VAST_UNRELAX_WARNINGS

namespace vast::core {
class FunctionOpInterface;

using function_op_interface = FunctionOpInterface;

namespace function_interface_impl {

/// Returns the dictionary attribute corresponding to the argument at 'index'.
/// If there are no argument attributes at 'index', a null attribute is
/// returned.
mlir::DictionaryAttr getArgAttrDict(FunctionOpInterface op, unsigned index);

/// Returns the dictionary attribute corresponding to the result at 'index'.
/// If there are no result attributes at 'index', a null attribute is
/// returned.
mlir::DictionaryAttr getResultAttrDict(FunctionOpInterface op, unsigned index);

/// Return all of the attributes for the argument at 'index'.
mlir::ArrayRef<mlir::NamedAttribute> getArgAttrs(FunctionOpInterface op, unsigned index);

/// Return all of the attributes for the result at 'index'.
mlir::ArrayRef<mlir::NamedAttribute> getResultAttrs(FunctionOpInterface op, unsigned index);

/// Set all of the argument or result attribute dictionaries for a function. The
/// size of `attrs` is expected to match the number of arguments/results of the
/// given `op`.
void setAllArgAttrDicts(FunctionOpInterface op, mlir::ArrayRef<mlir::DictionaryAttr> attrs);
void setAllArgAttrDicts(FunctionOpInterface op, mlir::ArrayRef<mlir::Attribute> attrs);
void setAllResultAttrDicts(FunctionOpInterface op,
                           mlir::ArrayRef<mlir::DictionaryAttr> attrs);
void setAllResultAttrDicts(FunctionOpInterface op, mlir::ArrayRef<mlir::Attribute> attrs);

/// Insert the specified arguments and update the function type attribute.
void insertFunctionArguments(FunctionOpInterface op,
                             mlir::ArrayRef<unsigned> argIndices, mlir::TypeRange argTypes,
                             mlir::ArrayRef<mlir::DictionaryAttr> argAttrs,
                             mlir::ArrayRef<mlir::Location> argLocs,
                             unsigned originalNumArgs, mlir::Type newType);

/// Insert the specified results and update the function type attribute.
void insertFunctionResults(FunctionOpInterface op,
                           mlir::ArrayRef<unsigned> resultIndices,
                           mlir::TypeRange resultTypes,
                           mlir::ArrayRef<mlir::DictionaryAttr> resultAttrs,
                           unsigned originalNumResults, mlir::Type newType);

/// Erase the specified arguments and update the function type attribute.
void eraseFunctionArguments(FunctionOpInterface op, const mlir::BitVector &argIndices,
                            mlir::Type newType);

/// Erase the specified results and update the function type attribute.
void eraseFunctionResults(FunctionOpInterface op,
                          const mlir::BitVector &resultIndices, mlir::Type newType);

/// Set a FunctionOpInterface operation's type signature.
void setFunctionType(FunctionOpInterface op, mlir::Type newType);

//===----------------------------------------------------------------------===//
// Function Argument mlir::Attribute.
//===----------------------------------------------------------------------===//

/// Set the attributes held by the argument at 'index'.
void setArgAttrs(FunctionOpInterface op, unsigned index,
                 mlir::ArrayRef<mlir::NamedAttribute> attributes);
void setArgAttrs(FunctionOpInterface op, unsigned index,
                 mlir::DictionaryAttr attributes);

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
template <typename ConcreteType>
void setArgAttr(ConcreteType op, unsigned index, mlir::StringAttr name,
                mlir::Attribute value) {
  mlir::NamedAttrList attributes(op.getArgAttrDict(index));
  mlir::Attribute oldValue = attributes.set(name, value);

  // If the attribute changed, then set the new arg attribute list.
  if (value != oldValue)
    op.setArgAttrs(index, attributes.getDictionary(value.getContext()));
}

/// Remove the attribute 'name' from the argument at 'index'. Returns the
/// removed attribute, or nullptr if `name` was not a valid attribute.
template <typename ConcreteType>
mlir::Attribute removeArgAttr(ConcreteType op, unsigned index, mlir::StringAttr name) {
  // Build an attribute list and remove the attribute at 'name'.
  mlir::NamedAttrList attributes(op.getArgAttrDict(index));
  mlir::Attribute removedAttr = attributes.erase(name);

  // If the attribute was removed, then update the argument dictionary.
  if (removedAttr)
    op.setArgAttrs(index, attributes.getDictionary(removedAttr.getContext()));
  return removedAttr;
}

//===----------------------------------------------------------------------===//
// Function Result mlir::Attribute.
//===----------------------------------------------------------------------===//

/// Set the attributes held by the result at 'index'.
void setResultAttrs(FunctionOpInterface op, unsigned index,
                    mlir::ArrayRef<mlir::NamedAttribute> attributes);
void setResultAttrs(FunctionOpInterface op, unsigned index,
                    mlir::DictionaryAttr attributes);

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
template <typename ConcreteType>
void setResultAttr(ConcreteType op, unsigned index, mlir::StringAttr name,
                   mlir::Attribute value) {
  mlir::NamedAttrList attributes(op.getResultAttrDict(index));
  mlir::Attribute oldAttr = attributes.set(name, value);

  // If the attribute changed, then set the new arg attribute list.
  if (oldAttr != value)
    op.setResultAttrs(index, attributes.getDictionary(value.getContext()));
}

/// Remove the attribute 'name' from the result at 'index'.
template <typename ConcreteType>
mlir::Attribute removeResultAttr(ConcreteType op, unsigned index, mlir::StringAttr name) {
  // Build an attribute list and remove the attribute at 'name'.
  mlir::NamedAttrList attributes(op.getResultAttrDict(index));
  mlir::Attribute removedAttr = attributes.erase(name);

  // If the attribute was removed, then update the result dictionary.
  if (removedAttr)
    op.setResultAttrs(index,
                      attributes.getDictionary(removedAttr.getContext()));
  return removedAttr;
}

/// This function defines the internal implementation of the `verifyTrait`
/// method on FunctionOpInterface::Trait.
template <typename ConcreteOp>
mlir::LogicalResult verifyTrait(ConcreteOp op) {
  if (failed(op.verifyType()))
    return llvm::failure();

  if (mlir::ArrayAttr allArgAttrs = op.getAllArgAttrs()) {
    unsigned numArgs = op.getNumArguments();
    if (allArgAttrs.size() != numArgs) {
      return op.emitOpError()
             << "expects argument attribute array to have the same number of "
                "elements as the number of function arguments, got "
             << allArgAttrs.size() << ", but expected " << numArgs;
    }
    for (unsigned i = 0; i != numArgs; ++i) {
      mlir::DictionaryAttr argAttrs =
          llvm::dyn_cast_or_null<mlir::DictionaryAttr>(allArgAttrs[i]);
      if (!argAttrs) {
        return op.emitOpError() << "expects argument attribute dictionary "
                                   "to be a mlir::DictionaryAttr, but got `"
                                << allArgAttrs[i] << "`";
      }

      // Verify that all of the argument attributes are dialect attributes, i.e.
      // that they contain a dialect prefix in their name.  Call the dialect, if
      // registered, to verify the attributes themselves.
      for (auto attr : argAttrs) {
        if (!attr.getName().strref().contains('.'))
          return op.emitOpError("arguments may only have dialect attributes");
        if (mlir::Dialect *dialect = attr.getNameDialect()) {
          if (failed(dialect->verifyRegionArgAttribute(op, /*regionIndex=*/0,
                                                       /*argIndex=*/i, attr)))
            return llvm::failure();
        }
      }
    }
  }
  if (mlir::ArrayAttr allResultAttrs = op.getAllResultAttrs()) {
    unsigned numResults = op.getNumResults();
    if (allResultAttrs.size() != numResults) {
      return op.emitOpError()
             << "expects result attribute array to have the same number of "
                "elements as the number of function results, got "
             << allResultAttrs.size() << ", but expected " << numResults;
    }
    for (unsigned i = 0; i != numResults; ++i) {
      mlir::DictionaryAttr resultAttrs =
          llvm::dyn_cast_or_null<mlir::DictionaryAttr>(allResultAttrs[i]);
      if (!resultAttrs) {
        return op.emitOpError() << "expects result attribute dictionary "
                                   "to be a mlir::DictionaryAttr, but got `"
                                << allResultAttrs[i] << "`";
      }

      // Verify that all of the result attributes are dialect attributes, i.e.
      // that they contain a dialect prefix in their name.  Call the dialect, if
      // registered, to verify the attributes themselves.
      for (auto attr : resultAttrs) {
        if (!attr.getName().strref().contains('.'))
          return op.emitOpError("results may only have dialect attributes");
        if (mlir::Dialect *dialect = attr.getNameDialect()) {
          if (failed(dialect->verifyRegionResultAttribute(op, /*regionIndex=*/0,
                                                          /*resultIndex=*/i,
                                                          attr)))
            return llvm::failure();
        }
      }
    }
  }

  // Check that the op has exactly one region for the body.
  if (op->getNumRegions() != 1)
    return op.emitOpError("expects one region");

  return op.verifyBody();
}
} // namespace function_interface_impl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Interface Declarations
//===----------------------------------------------------------------------===//

VAST_RELAX_WARNINGS
#include "vast/Dialect/Core/Interfaces/FunctionInterface.h.inc"
VAST_UNRELAX_WARNINGS

#endif // VAST_DIALECT_CORE_FUNCTIONINTERFACE_HPP
