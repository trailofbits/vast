// Copyright (c) 2022-present, Trail of Bits, Inc.

#include <vast/Dialect/HighLevel/HighLevelLinkage.hpp>

VAST_RELAX_WARNINGS
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::hl {

    using Visibility = mlir::SymbolTable::Visibility;

    static bool should_be_in_comdat(const clang::Decl *decl) {
        auto &actx = decl->getASTContext();
        auto triple = actx.getTargetInfo().getTriple();

        if (!triple.supportsCOMDAT()) {
            return false;
        }

        if (decl->hasAttr< clang::SelectAnyAttr >()) {
            return true;
        }

        clang::GVALinkage linkage = [&] {
            if (auto *var = dyn_cast< clang::VarDecl >(decl)) {
                return actx.GetGVALinkageForVariable(var);
            }
            return actx.GetGVALinkageForFunction(clang::cast< clang::FunctionDecl>(decl));
        } ();

        switch (linkage) {
            case clang::GVA_Internal:
            case clang::GVA_AvailableExternally:
            case clang::GVA_StrongExternal:
                return false;
            case clang::GVA_DiscardableODR:
            case clang::GVA_StrongODR:
                return true;
        }

        VAST_UNREACHABLE("No such linkage");
    }

    bool is_vardecl_strong_definition(const clang::VarDecl* decl) {
        auto &actx = decl->getASTContext();

        // TODO: auto nocommon = actx.getCodeGenOpts().NoCommon;
        bool nocommon = false;

        // Don't give variables common linkage if -fno-common was specified unless it
        // was overridden by a NoCommon attribute.
        if ((nocommon || decl->hasAttr< clang::NoCommonAttr >()) && !decl->hasAttr< clang::CommonAttr >()) {
            return true;
        }

        // C11 6.9.2/2:
        //   A declaration of an identifier for an object that has file scope without
        //   an initializer, and without a storage-class specifier or with the
        //   storage-class specifier static, constitutes a tentative definition.
        if (decl->getInit() || decl->hasExternalStorage()) {
            return true;
        }

        // A variable cannot be both common and exist in a section.
        if (decl->hasAttr< clang::SectionAttr >()) {
            return true;
        }

        // A variable cannot be both common and exist in a section.
        // We don't try to determine which is the right section in the front-end.
        // If no specialized section name is applicable, it will resort to default.
        if (decl->hasAttr< clang::PragmaClangBSSSectionAttr >() ||
            decl->hasAttr< clang::PragmaClangDataSectionAttr >() ||
            decl->hasAttr< clang::PragmaClangRelroSectionAttr >() ||
            decl->hasAttr< clang::PragmaClangRodataSectionAttr >())
        {
            return true;
        }

        // Thread local vars aren't considered common linkage.
        if (decl->getTLSKind()) {
            return true;
        }

        // Tentative definitions marked with WeakImportAttr are true definitions.
        if (decl->hasAttr< clang::WeakImportAttr >()) {
            return true;
        }

        // A variable cannot be both common and exist in a comdat.
        if (should_be_in_comdat(decl)) {
            return true;
        }

        // Declarations with a required alignment do not have common linkage in MSVC
        // mode.
        if (actx.getTargetInfo().getCXXABI().isMicrosoft()) {
            if (decl->hasAttr< clang::AlignedAttr >()) {
                return true;
            }

            auto type = decl->getType();
            if (actx.isAlignmentRequired(type)) {
                return true;
            }

            if (const auto *rty = type->getAs< clang::RecordType >()) {
                const clang::RecordDecl *rec = rty->getDecl();
                for (const auto *field : rec->fields()) {
                    if (field->isBitField())
                        continue;
                    if (field->hasAttr< clang::AlignedAttr >())
                        return true;
                    if (actx.isAlignmentRequired(field->getType()))
                        return true;
                }
            }
        }

        // Microsoft's link.exe doesn't support alignments greater than 32 bytes for
        // common symbols, so symbols with greater alignment requirements cannot be
        // common.
        // Other COFF linkers (ld.bfd and LLD) support arbitrary power-of-two
        // alignments for common symbols via the aligncomm directive, so this
        // restriction only applies to MSVC environments.
        if (actx.getTargetInfo().getTriple().isKnownWindowsMSVCEnvironment() &&
            actx.getTypeAlignIfKnown(decl->getType()) > actx.toBits(clang::CharUnits::fromQuantity(32)))
        {
            return true;
        }

        return false;
    }

    // adapted from getMLIRVisibilityFromCIRLinkage
    Visibility get_visibility_from_linkage(GlobalLinkageKind linkage) {
        switch (linkage) {
            case GlobalLinkageKind::InternalLinkage:
            case GlobalLinkageKind::PrivateLinkage:
                return Visibility::Private;
            case GlobalLinkageKind::ExternalLinkage:
            case GlobalLinkageKind::AvailableExternallyLinkage:
            case GlobalLinkageKind::ExternalWeakLinkage:
            case GlobalLinkageKind::LinkOnceODRLinkage:
                return Visibility::Public;
            default:
                VAST_UNREACHABLE("unsupported linkage kind {0}", stringifyGlobalLinkageKind(linkage));
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

        auto &actx = decl->getASTContext();
        const auto &opts = actx.getLangOpts();

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
            return !opts.AppleKext
                ? GlobalLinkageKind::LinkOnceODRLinkage
                : GlobalLinkageKind::InternalLinkage;
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
            if (opts.AppleKext) {
                return GlobalLinkageKind::ExternalLinkage;
            }

            if (opts.CUDA && opts.CUDAIsDevice && !opts.GPURelocatableDeviceCode) {
                return decl->hasAttr< clang::CUDAGlobalAttr >()
                    ? GlobalLinkageKind::ExternalLinkage
                    : GlobalLinkageKind::InternalLinkage;
            }

            return GlobalLinkageKind::WeakODRLinkage;
        }

        // C++ doesn't have tentative definitions and thus cannot have common
        // linkage.
        if (!opts.CPlusPlus) {
            if (auto var = clang::dyn_cast< clang::VarDecl >(decl)) {
                if (!is_vardecl_strong_definition(var)) {
                    return GlobalLinkageKind::CommonLinkage;
                }
            }
        }

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
        const auto *decl = clang::cast< clang::FunctionDecl >(glob.getDecl());

        auto &actx = decl->getASTContext();
        auto linkage = actx.GetGVALinkageForFunction(decl);

        if (auto ctor = clang::dyn_cast< clang::CXXConstructorDecl >(decl)) {
            if (ctor->isInheritingConstructor() && actx.getTargetInfo().getCXXABI().isMicrosoft()) {
                // Just like in LLVM codegen:
                // Our approach to inheriting constructors is fundamentally different from
                // that used by the MS ABI, so keep our inheriting constructor thunks
                // internal rather than trying to pick an unambiguous mangling for them.
                return GlobalLinkageKind::InternalLinkage;
            }
        }

        return get_declarator_linkage(decl, linkage, /* is const variable */ false);
    }

} // namespace vast::hl
