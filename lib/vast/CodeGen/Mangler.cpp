// Copyright (c) 2023, Trail of Bits, Inc.

#include <vast/CodeGen/Mangler.hpp>

namespace vast::cg
{
    mangled_name_ref CodeGenMangler::get_mangled_name(
        clang::GlobalDecl decl, const clang::TargetInfo &target_info, const std::string &module_name_hash
    ) {
        auto canonical = decl.getCanonicalDecl();

        // Some ABIs don't have constructor variants. Make sure that base and complete
        // constructors get mangled the same.
        if (const auto *ctor = clang::dyn_cast< clang::CXXConstructorDecl >(canonical.getDecl())) {
            VAST_UNIMPLEMENTED_IF(!target_info.getCXXABI().hasConstructorVariants());
        }

        // VAST_UNIMPLEMENTED_IF(!langOpts.CUDAIsDevice);

        // Keep the first result in the case of a mangling collision.
        auto mangled_name = mangle(decl, module_name_hash);

        auto result = manglings.insert(std::make_pair(mangled_name, decl));
        return mangled_decl_names[canonical] = mangled_name_ref{ result.first->first() };
    }

    std::optional< clang::GlobalDecl > CodeGenMangler::lookup_representative_decl(mangled_name_ref mangled_name) const {
        if (auto res = manglings.find(mangled_name.name); res != manglings.end()) {
            return res->getValue();
        }

        return std::nullopt;
    }

    // Returns true if decl is a function decl with internal linkage and needs a
    // unique suffix after the mangled name.
    static bool is_unique_internal_linkage_decl(clang::GlobalDecl /* decl */, const std::string &module_name_hash) {
        if (!module_name_hash.empty()) {
            VAST_UNIMPLEMENTED_MSG( "Unique internal linkage names NYI");
        }
        return false;
    }

    bool is_x86_regular(const clang::FunctionDecl *fn) {
        return fn && fn->getType()->castAs< clang::FunctionType >()->getCallConv() == clang::CC_X86RegCall;
    }

    bool is_cuda_kernel_name(const clang::FunctionDecl *fn, clang::GlobalDecl decl) {
        return fn
            && fn->hasAttr< clang::CUDAGlobalAttr >()
            && decl.getKernelReferenceKind() == clang::KernelReferenceKind::Stub;
    }

    std::string CodeGenMangler::mangle(clang::GlobalDecl decl, const std::string &module_name_hash) const {
        const auto *named = clang::cast< clang::NamedDecl >(decl.getDecl());

        llvm::SmallString< 256 > buffer;
        llvm::raw_svector_ostream out(buffer);

        if (!module_name_hash.empty()) {
            VAST_UNIMPLEMENTED_MSG("mangling with uninitilized module");
        }

        if (mangle_context->shouldMangleDeclName(named)) {
            mangle_context->mangleName(decl.getWithDecl(named), out);
        } else {
            auto *identifier = named->getIdentifier();
            VAST_CHECK(identifier, "Attempt to mangle unnamed decl.");

            const auto *fn = clang::dyn_cast< clang::FunctionDecl >(named);

            if (is_x86_regular(fn)) {
                VAST_UNIMPLEMENTED_MSG("x86 function name mangling");
            } else if (is_cuda_kernel_name(fn, decl)) {
                VAST_UNIMPLEMENTED_MSG("cuda name mangling");
            } else {
                out << identifier->getName();
            }
        }

        // Check if the module name hash should be appended for internal linkage
        // symbols. This should come before multi-version target suffixes are
        // appendded. This is to keep the name and module hash suffix of the internal
        // linkage function together. The unique suffix should only be added when name
        // mangling is done to make sure that the final name can be properly
        // demangled. For example, for C functions without prototypes, name mangling
        // is not done and the unique suffix should not be appended then.
        VAST_UNIMPLEMENTED_IF(is_unique_internal_linkage_decl(decl, module_name_hash));

        if (const auto *fn = clang::dyn_cast< clang::FunctionDecl >(named)) {
            VAST_UNIMPLEMENTED_IF(fn->isMultiVersion());
        }

        // VAST_UNIMPLEMENTED_IF(CGM.getLangOpts().GPURelocatableDeviceCode);

        return std::string(out.str());
    }
} // namespace vast::cg


namespace llvm {

    [[nodiscard]] hash_code hash_value(vast::cg::mangled_name_ref mangled) {
        return hash_value(mangled.name);
    }

} // llvm
