// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenStmtVisitor.hpp"

namespace vast::hl
{
    CastKind cast_kind(const clang::CastExpr *expr)
    {
        switch (expr->getCastKind()) {
            case clang::CastKind::CK_Dependent: return CastKind::Dependent;
            case clang::CastKind::CK_BitCast: return CastKind::BitCast;
            case clang::CastKind::CK_LValueBitCast: return CastKind::LValueBitCast;
            case clang::CastKind::CK_LValueToRValueBitCast: return CastKind::LValueToRValueBitCast;
            case clang::CastKind::CK_LValueToRValue: return CastKind::LValueToRValue;
            case clang::CastKind::CK_NoOp: return CastKind::NoOp;

            case clang::CastKind::CK_BaseToDerived: return CastKind::BaseToDerived;
            case clang::CastKind::CK_DerivedToBase: return CastKind::DerivedToBase;
            case clang::CastKind::CK_UncheckedDerivedToBase: return CastKind::UncheckedDerivedToBase;
            case clang::CastKind::CK_Dynamic: return CastKind::Dynamic;
            case clang::CastKind::CK_ToUnion: return CastKind::ToUnion;

            case clang::CastKind::CK_ArrayToPointerDecay: return CastKind::ArrayToPointerDecay;
            case clang::CastKind::CK_FunctionToPointerDecay: return CastKind::FunctionToPointerDecay;
            case clang::CastKind::CK_NullToPointer: return CastKind::NullToPointer;
            case clang::CastKind::CK_NullToMemberPointer: return CastKind::NullToMemberPointer;
            case clang::CastKind::CK_BaseToDerivedMemberPointer: return CastKind::BaseToDerivedMemberPointer;
            case clang::CastKind::CK_DerivedToBaseMemberPointer: return CastKind::DerivedToBaseMemberPointer;
            case clang::CastKind::CK_MemberPointerToBoolean: return CastKind::MemberPointerToBoolean;
            case clang::CastKind::CK_ReinterpretMemberPointer: return CastKind::ReinterpretMemberPointer;
            case clang::CastKind::CK_UserDefinedConversion: return CastKind::UserDefinedConversion;
            case clang::CastKind::CK_ConstructorConversion: return CastKind::ConstructorConversion;

            case clang::CastKind::CK_IntegralToPointer: return CastKind::IntegralToPointer;
            case clang::CastKind::CK_PointerToIntegral: return CastKind::PointerToIntegral;
            case clang::CastKind::CK_PointerToBoolean : return CastKind::PointerToBoolean;

            case clang::CastKind::CK_ToVoid: return CastKind::ToVoid;
            case clang::CastKind::CK_VectorSplat: return CastKind::VectorSplat;

            case clang::CastKind::CK_IntegralCast: return CastKind::IntegralCast;
            case clang::CastKind::CK_IntegralToBoolean: return CastKind::IntegralToBoolean;
            case clang::CastKind::CK_IntegralToFloating: return CastKind::IntegralToFloating;
            case clang::CastKind::CK_FloatingToFixedPoint: return CastKind::FloatingToFixedPoint;
            case clang::CastKind::CK_FixedPointToFloating: return CastKind::FixedPointToFloating;
            case clang::CastKind::CK_FixedPointCast: return CastKind::FixedPointCast;
            case clang::CastKind::CK_FixedPointToIntegral: return CastKind::FixedPointToIntegral;
            case clang::CastKind::CK_IntegralToFixedPoint: return CastKind::IntegralToFixedPoint;
            case clang::CastKind::CK_FixedPointToBoolean: return CastKind::FixedPointToBoolean;
            case clang::CastKind::CK_FloatingToIntegral: return CastKind::FloatingToIntegral;
            case clang::CastKind::CK_FloatingToBoolean: return CastKind::FloatingToBoolean;
            case clang::CastKind::CK_BooleanToSignedIntegral: return CastKind::BooleanToSignedIntegral;
            case clang::CastKind::CK_FloatingCast: return CastKind::FloatingCast;

            case clang::CastKind::CK_CPointerToObjCPointerCast: return CastKind::CPointerToObjCPointerCast;
            case clang::CastKind::CK_BlockPointerToObjCPointerCast: return CastKind::BlockPointerToObjCPointerCast;
            case clang::CastKind::CK_AnyPointerToBlockPointerCast: return CastKind::AnyPointerToBlockPointerCast;
            case clang::CastKind::CK_ObjCObjectLValueCast: return CastKind::ObjCObjectLValueCast;

            case clang::CastKind::CK_FloatingRealToComplex: return CastKind::FloatingRealToComplex;
            case clang::CastKind::CK_FloatingComplexToReal: return CastKind::FloatingComplexToReal;
            case clang::CastKind::CK_FloatingComplexToBoolean: return CastKind::FloatingComplexToBoolean;
            case clang::CastKind::CK_FloatingComplexCast: return CastKind::FloatingComplexCast;
            case clang::CastKind::CK_FloatingComplexToIntegralComplex: return CastKind::FloatingComplexToIntegralComplex;
            case clang::CastKind::CK_IntegralRealToComplex: return CastKind::IntegralRealToComplex;
            case clang::CastKind::CK_IntegralComplexToReal: return CastKind::IntegralComplexToReal;
            case clang::CastKind::CK_IntegralComplexToBoolean: return CastKind::IntegralComplexToBoolean;
            case clang::CastKind::CK_IntegralComplexCast: return CastKind::IntegralComplexCast;
            case clang::CastKind::CK_IntegralComplexToFloatingComplex: return CastKind::IntegralComplexToFloatingComplex;

            case clang::CastKind::CK_ARCProduceObject: return CastKind::ARCProduceObject;
            case clang::CastKind::CK_ARCConsumeObject: return CastKind::ARCConsumeObject;
            case clang::CastKind::CK_ARCReclaimReturnedObject: return CastKind::ARCReclaimReturnedObject;
            case clang::CastKind::CK_ARCExtendBlockObject: return CastKind::ARCExtendBlockObject;

            case clang::CastKind::CK_AtomicToNonAtomic: return CastKind::AtomicToNonAtomic;
            case clang::CastKind::CK_NonAtomicToAtomic: return CastKind::NonAtomicToAtomic;

            case clang::CastKind::CK_CopyAndAutoreleaseBlockObject: return CastKind::CopyAndAutoreleaseBlockObject;
            case clang::CastKind::CK_BuiltinFnToFnPtr: return CastKind::BuiltinFnToFnPtr;

            case clang::CastKind::CK_ZeroToOCLOpaqueType: return CastKind::ZeroToOCLOpaqueType;
            case clang::CastKind::CK_AddressSpaceConversion: return CastKind::AddressSpaceConversion;
            case clang::CastKind::CK_IntToOCLSampler: return CastKind::IntToOCLSampler;

            case clang::CastKind::CK_MatrixCast: return CastKind::MatrixCast;
        }

        VAST_UNREACHABLE( "unsupported cast kind" );
    }

    IdentKind ident_kind(const clang::PredefinedExpr *expr)
    {
        switch(expr->getIdentKind())
        {
            case clang::PredefinedExpr::IdentKind::Func : return IdentKind::Func;
            case clang::PredefinedExpr::IdentKind::Function : return IdentKind::Function;
            case clang::PredefinedExpr::IdentKind::LFunction : return IdentKind::LFunction;
            case clang::PredefinedExpr::IdentKind::FuncDName : return IdentKind::FuncDName;
            case clang::PredefinedExpr::IdentKind::FuncSig : return IdentKind::FuncSig;
            case clang::PredefinedExpr::IdentKind::LFuncSig : return IdentKind::LFuncSig;
            case clang::PredefinedExpr::IdentKind::PrettyFunction :
                return IdentKind::PrettyFunction;
            case clang::PredefinedExpr::IdentKind::PrettyFunctionNoVirtual :
                return IdentKind::PrettyFunctionNoVirtual;
        }
    }

} // namespace vast::hl
