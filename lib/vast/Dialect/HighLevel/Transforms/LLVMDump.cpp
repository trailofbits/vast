// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/PatternApplicator.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>

#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/raw_ostream.h>
VAST_UNRELAX_WARNINGS


#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>
#include <vast/Dialect/HighLevel/HighLevelUtils.hpp>

#include "PassesDetails.hpp"

#include <unordered_map>
#include <unordered_set>

namespace vast::hl
{
    class ToLLLVMIR : public mlir::LLVMTranslationDialectInterface
    {
      public:
        using Base = mlir::LLVMTranslationDialectInterface;
        using Base::Base;

        mlir::LogicalResult convertOperation(mlir::Operation *op, llvm::IRBuilderBase &irb,
                                             mlir::LLVM::ModuleTranslation &state) const final
        {
            return llvm::TypeSwitch< mlir::Operation *, mlir::LogicalResult >(op)
                .Case([&](hl::TypeDefOp) {
                    return mlir::success();
                })
                .Default([&](mlir::Operation *) {
                    return mlir::failure();
                });
        }
    };

    using mlir_to_llvm = std::unordered_map< mlir::Operation *, llvm::Instruction * >;
        // op->getStringRef.

    struct get_mapping
    {
        // func to mapping of its things
        std::map< std::string, mlir_to_llvm > function_mapping;
        std::map< operation, llvm::GlobalVariable * > global_vars;
        std::map< operation, llvm::Function * > global_functions;

        inline static const std::unordered_set< std::string > m_skip =
        {
            "llvm.mlir.constant"
        };

        inline static const std::unordered_set< std::string > allowed_miss =
        {
            "llvm.bitcast",
            "llvm.mlir.addressof"
        };

        inline static const std::unordered_set< std::string > l_skip =
        {
        };

        inline static const std::unordered_map< std::string, std::string > translations =
        {
            { "llvm.alloca", "alloca" },
            { "llvm.bitcast","bitcast" },
            { "llvm.store", "store" },
            { "llvm.load", "load" },
            { "llvm.return", "ret" },
            { "llvm.call", "call" },
            { "llvm.getelementptr", "getelementptr" },

            { "llvm.mlir.addressof", "" }
        };

        std::tuple< mlir::LLVM::LLVMFuncOp, llvm::Function * >
        get_fn( auto name, auto m_mod, auto l_mod )
        {
            auto l_fn = l_mod->getFunction(name);
            for (auto m_fn : top_level_ops< mlir::LLVM::LLVMFuncOp >(m_mod))
            {
                if (m_fn.getName() == name)
                    return { m_fn, l_fn };
            }

            return {};
        }

        auto key(mlir::Block::iterator it) -> std::string
        {
            return it->getName().getStringRef().str();
        }

        auto key(llvm::BasicBlock::iterator it) -> std::string
        {
            return it->getOpcodeName();
        }


        bool skip(mlir::Block::iterator it) { return m_skip.count( key( it ) ); }
        bool skip(llvm::BasicBlock::iterator it) { return l_skip.count( key( it ) ); }

        bool match( auto m_it, auto l_it )
        {
            auto m = translations.find( key( m_it ) );
            VAST_CHECK( m != translations.end(),
                        "Missing translation {0}: {1}", key( m_it ),  *m_it );
            return m->second == key(l_it);
        }

        auto annotate_functions(mlir::ModuleOp op, llvm::Module *l_mod)
        {
            auto [ m_func, l_func ] = get_fn("main", op, l_mod);
            VAST_ASSERT( m_func && l_func );
            global_functions.emplace( m_func, l_func );
            llvm::errs() << "Matched functions!\n";


            auto &current = function_mapping[ "main" ];

            auto m_it = m_func.getRegion().begin()->begin();
            auto m_end = m_func.getRegion().begin()->end();
            llvm::BasicBlock::iterator l_it = l_func->begin()->begin();
            llvm::BasicBlock::iterator l_end = l_func->begin()->end();

            while (m_it != m_end)
            {
                if (skip(m_it))
                {
                    llvm::errs() << "m_skip rule: " << key(m_it) << "\n";
                    ++m_it;
                    continue;
                }
                VAST_ASSERT(l_it != l_end);

                llvm::errs() << "Matching: " << key(m_it) << " to " << key(l_it) << "\n";

                if (!match(m_it, l_it))
                {
                    if (skip(l_it))
                    {
                        llvm::errs() << " \tl_skip rule: " << key(l_it) << "\n";
                        ++l_it;
                        continue;
                    }
                    if ( allowed_miss.count( key( m_it ) ) )
                    {
                        llvm::errs() << " \tallowed_miss rule: " << key(m_it) << "\n";
                        ++m_it;
                        continue;
                    }
                    VAST_CHECK(false, "Cannot progress on {0}!", key(m_it));
                }
                llvm::errs() << ".... Matched!\n";

                current.emplace( &*m_it, &*l_it );
                ++l_it;
                ++m_it;
            }
        }

        auto annotate_gvs(mlir::ModuleOp m_mod, llvm::Module *l_mod)
        {
            for (auto m_var : top_level_ops< mlir::LLVM::GlobalOp >(m_mod))
            {
                auto l_var = l_mod->getGlobalVariable(m_var.getName());
                global_vars.emplace( m_var, l_var );
                llvm::errs() << "Matched globals!\n";
            }
        }

        void get(mlir::ModuleOp m_mod, llvm::Module *l_mod)
        {
            // Not sure what to do with global and functions yet.
            annotate_functions(m_mod, l_mod);
            annotate_gvs(m_mod, l_mod);
            //auto print = [&]( operation thing )
            //{
            //    llvm::errs() << thing->getName().getStringRef() << "\n";
            //};
            //m_mod->walk< mlir::WalkOrder::PostOrder >(print);
        }
    };

    struct LLVMDump : LLVMDumpBase< LLVMDump >
    {
        void runOnOperation() override;
    };

    void LLVMDump::runOnOperation()
    {
        auto &mctx = this->getContext();
        mlir::ModuleOp op = this->getOperation();

        // If the old data layout with high level types is left in the module,
        // some parsing functionality inside the `mlir::translateModuleToLLVMIR`
        // will fail and no conversion translation happens, even in case these
        // entries are not used at all.
        auto old_dl = op->getAttr(mlir::DLTIDialect::kDataLayoutAttrName);
        op->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                    mlir::DataLayoutSpecAttr::get(&mctx, {}));

        llvm::LLVMContext lctx;
        auto lmodule = mlir::translateModuleToLLVMIR(op, lctx);
        if (!lmodule)
            return signalPassFailure();

        // Restore the data layout in case this module is getting re-used later.
        op->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, old_dl);

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        mlir::ExecutionEngine::setupTargetTriple(lmodule.get());

        auto dump = [&](auto &stream)
        {
            stream << *lmodule;
            stream.flush();
        };

        (void)get_mapping{}.get(op, lmodule.get());

        auto outname = this->bitcode_file.getValue();
        if (outname.empty())
            return dump(llvm::outs());

        std::error_code ec;
        llvm::raw_fd_ostream out(outname, ec);

        VAST_CHECK(!ec, "Cannot store bitcode: {0}", ec.message());
        dump(out);
    }
} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createLLVMDumpPass()
{
    return std::make_unique< LLVMDump >();
}
