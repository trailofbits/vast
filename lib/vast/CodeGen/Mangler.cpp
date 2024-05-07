// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/CodeGen/Mangler.hpp"

#include "vast/Util/Maybe.hpp"

namespace vast::cg
{
    std::optional< symbol_name > default_symbol_mangler::symbol(clang_global decl) {
        return Maybe(decl.getCanonicalDecl().getDecl())
            .and_then(dyn_cast< clang_named_decl >)
            .and_then([&](auto decl) {
                return symbol(decl);
            })
            .take();
    }


    std::optional< symbol_name > default_symbol_mangler::symbol(const clang_named_decl *decl) {
        auto &actx = mangle_context->getASTContext();

        if (mangled_decl_names.count(decl)) {
            return mangled_decl_names[decl];
        }

        // Some ABIs don't have constructor variants. Make sure that base and
        // complete constructors get mangled the same.
        if (const auto *ctor = clang::dyn_cast< clang::CXXConstructorDecl >(decl)) {
            if (!actx.getTargetInfo().getCXXABI().hasConstructorVariants()) {
                return std::nullopt;
            }
        }

        // Keep the first result in the case of a mangling collision.
        if (auto mangled_name = mangle(decl)) {
            auto [it, _] = manglings.try_emplace(mangled_name.value(), decl);
            return mangled_decl_names[decl] = it->getKey();
        }

        return std::nullopt;
    }

    std::optional< symbol_name > default_symbol_mangler::symbol(const clang_decl_ref_expr *decl) {
        return Maybe(decl->getDecl())
            .and_then([&](auto decl) {
                return symbol(decl);
            })
            .take();
    }

    // Returns true if decl is a function decl with internal linkage and needs a
    // unique suffix after the mangled name.
    static bool is_unique_internal_linkage_decl(const clang_named_decl */* decl */, const std::string &module_name_hash) {
        if (!module_name_hash.empty()) {
            // FIXME: Implement unique internal linkage names.
            return true;
        }
        return false;
    }

    bool is_x86_regular(const clang::FunctionDecl *fn) {
        return fn && fn->getType()->castAs< clang::FunctionType >()->getCallConv() == clang::CC_X86RegCall;
    }

    bool is_cuda_kernel_name(const clang::FunctionDecl *fn, clang_global decl) {
        return fn
            && fn->hasAttr< clang::CUDAGlobalAttr >()
            && decl.getKernelReferenceKind() == clang::KernelReferenceKind::Stub;
    }

    std::optional< std::string > default_symbol_mangler::mangle(const clang_named_decl *decl) {
        llvm::SmallString< 256 > buffer;
        llvm::raw_svector_ostream out(buffer);

        if (!module_name_hash.empty()) {
            return std::nullopt; // mangling with uninitilized module
        }

        auto anonoymous_mangle = [&]() {
            return "anonymous[" + std::to_string(decl->getID()) + "]";
        };

        if (const auto *field = clang::dyn_cast< clang::FieldDecl >(decl)) {
            if (field->isUnnamedBitfield() || field->isAnonymousStructOrUnion()) {
                return anonoymous_mangle();
            }
        }

        if (const auto *record = clang::dyn_cast< clang::RecordDecl >(decl)) {
            if (record->isAnonymousStructOrUnion()) {
                return anonoymous_mangle();
            }
        }

        if (const auto *enum_decl = clang::dyn_cast< clang::NamedDecl >(decl)) {
            if (!enum_decl->getIdentifier()) {
                return anonoymous_mangle();
            }
        }

        auto should_mangle_filter = [](const clang_named_decl *decl) {
            if (auto var = clang::dyn_cast< clang::VarDecl >(decl)) {
                return !var->isLocalVarDeclOrParm();
            }

            return true; // check to see if decl should be mangled
        };


        if (should_mangle_filter(decl) && mangle_context->shouldMangleDeclName(decl)) {
            mangle_context->mangleName(decl, out);
        } else {
            auto *identifier = decl->getIdentifier();
            if (!identifier) {
                return std::nullopt; // attempt mangling of unnamed decl
            }

            // Check if the module name hash should be appended for internal linkage
            // symbols. This should come before multi-version target suffixes are
            // appendded. This is to keep the name and module hash suffix of the internal
            // linkage function together. The unique suffix should only be added when name
            // mangling is done to make sure that the final name can be properly
            // demangled. For example, for C functions without prototypes, name mangling
            // is not done and the unique suffix should not be appended then.
            if (is_unique_internal_linkage_decl(decl, module_name_hash)) {
                return std::nullopt; // unimplemented unique internal linkage names
            }

            if (const auto *fn = clang::dyn_cast< clang::FunctionDecl >(decl)) {
                if (fn->isMultiVersion()) {
                    return std::nullopt; // unimplemented multi-version function name mangling
                }

                if (is_x86_regular(fn)) {
                    return std::nullopt; // unimplemented x86 function name mangling
                } else if (is_cuda_kernel_name(fn, decl)) {
                    return std::nullopt; // unimplemented cuda name mangling
                } else {
                    out << identifier->getName();
                }
            } else {
                out << decl->getName();
            }
        }

        auto &actx = mangle_context->getASTContext();
        if (actx.getLangOpts().GPURelocatableDeviceCode) {
            return std::nullopt; // unimplemented GPURelocatableDeviceCode name mangling
        }

        return std::string(out.str());
    }
} // namespace vast::cg
