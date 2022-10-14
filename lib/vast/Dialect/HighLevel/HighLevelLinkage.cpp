// Copyright (c) 2022-present, Trail of Bits, Inc.

#include <vast/Dialect/HighLevel/HighLevelLinkage.hpp>

namespace vast::hl {

    using Visibility = mlir::SymbolTable::Visibility;

    Visibility get_visibility_from_linkage(GlobalLinkageKind linkage) {
        switch (linkage) {
            case GlobalLinkageKind::InternalLinkage:
            case GlobalLinkageKind::PrivateLinkage:
                return Visibility::Private;
            case GlobalLinkageKind::ExternalLinkage:
            case GlobalLinkageKind::ExternalWeakLinkage:
            case GlobalLinkageKind::LinkOnceODRLinkage:
                return Visibility::Public;
            default:
                VAST_UNREACHABLE("unsupported linkage kind {}",
                    stringifyGlobalLinkageKind(linkage)
                );
        }

        VAST_UNREACHABLE("missed linkage kind");
    }

    GlobalLinkageKind get_declarator_linkage(
        const clang::DeclaratorDecl *decl, clang::GVALinkage linkage, bool is_constant
    ) {
        if (linkage == clang::GVA_Internal) {
            return GlobalLinkageKind::InternalLinkage;
        }

        if (decl->hasAttr< clang::WeakAttr >()) {
            if (is_constant)
                return GlobalLinkageKind::WeakODRLinkage;
            else
                return GlobalLinkageKind::WeakAnyLinkage;
        }

        if (const auto *fn = decl->getAsFunction()) {
            if (fn->isMultiVersion() && linkage == clang::GVA_AvailableExternally) {
                return GlobalLinkageKind::LinkOnceAnyLinkage;
            }
        }

        // We are guaranteed to have a strong definition somewhere else,
        // so we can use available_externally linkage.
        if (linkage == clang::GVA_AvailableExternally) {
            return GlobalLinkageKind::AvailableExternallyLinkage;
        }

        // Note that Apple's kernel linker doesn't support symbol
        // coalescing, so we need to avoid linkonce and weak linkages there.
        // Normally, this means we just map to internal, but for explicit
        // instantiations we'll map to external.

        // In C++, the compiler has to emit a definition in every translation unit
        // that references the function.  We should use linkonce_odr because
        // a) if all references in this translation unit are optimized away, we
        // don't need to codegen it.  b) if the function persists, it needs to be
        // merged with other definitions. c) C++ has the ODR, so we know the
        // definition is dependable.
        if (linkage == clang::GVA_DiscardableODR) {
            VAST_UNIMPLEMENTED;
            // return !actx.getLangOpts().AppleKext
            //         ? GlobalLinkageKind::LinkOnceODRLinkage
            //         : GlobalLinkageKind::InternalLinkage;
        }

        // An explicit instantiation of a template has weak linkage, since
        // explicit instantiations can occur in multiple translation units
        // and must all be equivalent. However, we are not allowed to
        // throw away these explicit instantiations.
        //
        // CUDA/HIP: For -fno-gpu-rdc case, device code is limited to one TU,
        // so say that CUDA templates are either external (for kernels) or internal.
        // This lets llvm perform aggressive inter-procedural optimizations. For
        // -fgpu-rdc case, device function calls across multiple TU's are allowed,
        // therefore we need to follow the normal linkage paradigm.
        if (linkage == clang::GVA_StrongODR) {
            VAST_UNIMPLEMENTED;
            // if (getLangOpts().AppleKext)
            //     return GlobalLinkageKind::ExternalLinkage;
            // if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice &&
            //     !getLangOpts().GPURelocatableDeviceCode)
            // return D->hasAttr<CUDAGlobalAttr>()
            //             ? GlobalLinkageKind::ExternalLinkage
            //             : GlobalLinkageKind::InternalLinkage;
            // return GlobalLinkageKind::WeakODRLinkage;
        }

        // C++ doesn't have tentative definitions and thus cannot have common
        // linkage.

        // TODO:
        // if (!getLangOpts().CPlusPlus && isa<VarDecl>(D) &&
        //     !isVarDeclStrongDefinition(astCtx, *this, cast<VarDecl>(D),
        //                                 getCodeGenOpts().NoCommon))
        // {
        //     return GlobalLinkageKind::CommonLinkage;
        // }

        // selectany symbols are externally visible, so use weak instead of
        // linkonce.  MSVC optimizes away references to const selectany globals, so
        // all definitions should be the same and ODR linkage should be used.
        // http://msdn.microsoft.com/en-us/library/5tkz6s71.aspx
        if (decl->hasAttr< clang::SelectAnyAttr >()) {
            return GlobalLinkageKind::WeakODRLinkage;
        }

        // Otherwise, we have strong external linkage.
        VAST_ASSERT(linkage == clang::GVA_StrongExternal);
        return GlobalLinkageKind::ExternalLinkage;
    }

    GlobalLinkageKind get_function_linkage(clang::GlobalDecl glob) {
        // TODO:
        // const auto *D = cast<FunctionDecl>(GD.getDecl());

        // GVALinkage Linkage = astCtx.GetGVALinkageForFunction(D);

        // if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(D))
        //     assert(0 && "NYI");

        // if (isa<CXXConstructorDecl>(D) &&
        //     cast<CXXConstructorDecl>(D)->isInheritingConstructor() &&
        //     astCtx.getTargetInfo().getCXXABI().isMicrosoft()) {
        //     // Just like in LLVM codegen:
        //     // Our approach to inheriting constructors is fundamentally different from
        //     // that used by the MS ABI, so keep our inheriting constructor thunks
        //     // internal rather than trying to pick an unambiguous mangling for them.
        //     return mlir::cir::GlobalLinkageKind::InternalLinkage;
        // }

        // return getCIRLinkageForDeclarator(D, Linkage, /*IsConstantVariable=*/false);
        return GlobalLinkageKind::ExternalLinkage;
    }

} // namespace vast::hl
