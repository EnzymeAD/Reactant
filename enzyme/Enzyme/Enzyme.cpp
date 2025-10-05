//===- Enzyme.cpp - Automatic Differentiation Transformation Pass  -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains Enzyme, a transformation pass that takes replaces calls
// to function calls to *__enzyme_autodiff* with a call to the derivative of
// the function passed as the first argument.
//
//===----------------------------------------------------------------------===//
#define private public
#include "llvm/IR/Module.h"
#undef private
#include <llvm/Config/llvm-config.h>
#include <memory>

#include "llvm/ADT/StringRef.h"
#include <dlfcn.h>

#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/PointerLikeTypeTraits.h>

#if LLVM_VERSION_MAJOR >= 16
#define private public
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#undef private
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <optional>
#if LLVM_VERSION_MAJOR <= 16
#include "llvm/ADT/Optional.h"
#endif
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Transforms/IPO/GlobalOpt.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"

#include "PreserveNVVM.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-reactant-intrinsic"

llvm::cl::opt<std::string>
    Passes("raising-plugin-path", cl::init(""), cl::Hidden,
           cl::desc("Print before and after fns for autodiff"));

llvm::cl::opt<std::string>
    ReactantBackend("reactant-backend", cl::init("cuda"), cl::Hidden,
                    cl::desc("Default backend for reactant"));

llvm::cl::opt<std::string>
    DeviceLibraries("reactant-device-lib", cl::Hidden, cl::init(""),
                    cl::desc("Library to link during device compilation"));

namespace {

constexpr char cudaLaunchSymbolName[] = "cudaLaunchKernel";

constexpr char cudaPushConfigName[] = "__cudaPushCallConfiguration";
constexpr char cudaPopConfigName[] = "__cudaPopCallConfiguration";

SmallVector<CallBase *> gatherCallers(Function *F) {
  if (!F)
    return {};
  SmallVector<CallBase *> ToHandle;
  for (auto User : F->users())
    if (auto CI = dyn_cast<CallBase>(User))
      if (CI->getCalledFunction() == F)
        ToHandle.push_back(CI);
  return ToHandle;
}

void fixup(Module &M) {

  auto LaunchKernelFunc = M.getFunction(cudaLaunchSymbolName);
  if (!LaunchKernelFunc)
    return;

  SmallPtrSet<Function *, 8> CoercedKernels;
  for (CallBase *CI : gatherCallers(LaunchKernelFunc)) {
    IRBuilder<> Builder(CI);
    auto GridDim1 = CI->getArgOperand(1);
    auto GridDim2 = CI->getArgOperand(2);
    auto BlockDim1 = CI->getArgOperand(3);
    auto BlockDim2 = CI->getArgOperand(4);
    auto ArgPtr = CI->getArgOperand(5);
    auto SharedMemSize = CI->getArgOperand(6);
    auto StreamPtr = CI->getArgOperand(7);
    
    auto StubFunc = cast<Function>(CI->getArgOperand(0));

    if (StubFunc->getName().starts_with("reactant$"))
      continue;
    auto NewFn = M.getOrInsertFunction(("reactant$" + StubFunc->getName()).str(), StubFunc->getFunctionType(), StubFunc->getAttributes());
    
    auto GridDimX = Builder.CreateTrunc(GridDim1, Builder.getInt32Ty());
    auto GridDimY = Builder.CreateLShr(
        GridDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    GridDimY = Builder.CreateTrunc(GridDimY, Builder.getInt32Ty());
    auto GridDimZ = GridDim2;
    auto BlockDimX = Builder.CreateTrunc(BlockDim1, Builder.getInt32Ty());
    auto BlockDimY = Builder.CreateLShr(
        BlockDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    BlockDimY = Builder.CreateTrunc(BlockDimY, Builder.getInt32Ty());
    auto BlockDimZ = BlockDim2;
    
    Value * Args[] = {
        NewFn.getCallee(),   GridDimX,  GridDimY,      GridDimZ,  BlockDimX,
        BlockDimY, BlockDimZ, SharedMemSize, StreamPtr,  ArgPtr
    };
    
    SmallVector<Type *> ArgTypes;
    for (Value *V : Args)
      ArgTypes.push_back(V->getType());
    auto MlirLaunchFunc = M.getOrInsertFunction(
        "__mlir_cuda_caller_phase2",
        FunctionType::get(Type::getVoidTy(M.getContext()), {}, true));

    Builder.CreateCall(MlirLaunchFunc, Args);
    CoercedKernels.insert(StubFunc);
    if (auto II = dyn_cast<InvokeInst>(CI)) {
      Builder.CreateBr(II->getNormalDest());
      II->getUnwindDest()->removePredecessor(II->getParent());
    }
    CI->eraseFromParent();
  }

  for (auto StubFunc : CoercedKernels) {
    for (CallBase *CI : gatherCallers(StubFunc)) {
       InlineFunctionInfo IFI;
          InlineResult Res =
              InlineFunction(*CI, IFI, /*MergeAttributes=*/false);
          assert(Res.isSuccess());
    }
  }

  // Map of runtime function, index of the entry fn
  std::pair<const char *, int> runtime_fns[] = {
      {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", 1},
      {"cudaFuncGetAttributes", 1},
      {"cudaFuncGetName", 1},
      {"cudaFuncSetAttribute", 0},
      {"cudaFuncSetCacheConfig", 0},
  };
  for (auto &pair : runtime_fns)
    if (auto occupancy = M.getFunction(pair.first)) {
      for (CallBase *CI : gatherCallers(occupancy)) {
        auto StubFunc = dyn_cast<Function>(CI->getArgOperand(pair.second));
        if (!StubFunc) {
          llvm::errs() << " cuda runtime function " << pair.first
                       << " did not have constant argument, had "
                       << *CI->getArgOperand(pair.second)
                       << " errors may occur\n";
          continue;
        }
        if (StubFunc->getName().starts_with("reactant$"))
          continue;
        auto NewFn = M.getOrInsertFunction(
            ("reactant$" + StubFunc->getName()).str(),
            StubFunc->getFunctionType(), StubFunc->getAttributes());
        CI->setArgOperand(1, NewFn.getCallee());
      }
    }

  DenseMap<Function *, SmallVector<AllocaInst *, 6>> FuncAllocas;
  if (auto PushConfigFunc = M.getFunction(cudaPushConfigName)) {
  for (CallBase *CI : gatherCallers(PushConfigFunc)) {
    Function *TheFunc = CI->getFunction();
    IRBuilder<> IRB(&TheFunc->getEntryBlock(),
                    TheFunc->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    auto Allocas = FuncAllocas.lookup(TheFunc);
    if (Allocas.empty()) {
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "griddim64"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt32Ty(), nullptr, "griddim32"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "blockdim64"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt32Ty(), nullptr, "blockdim32"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "shmem_size"));
      Allocas.push_back(IRB.CreateAlloca(IRB.getPtrTy(), nullptr, "stream"));
      FuncAllocas.insert_or_assign(TheFunc, Allocas);
    }
    IRB.SetInsertPoint(CI);
    if (CI->arg_size() != Allocas.size()) {
      llvm::errs() << " size mismatch on: " << *CI << "\n";
    }
    for (auto [Arg, Alloca] : llvm::zip_equal(CI->args(), Allocas))
      IRB.CreateStore(Arg, Alloca);
    CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    CI->eraseFromParent();
  }
  }

  if (  auto PopConfigFunc = M.getFunction(cudaPopConfigName)) {
  for (CallBase *PopCall : gatherCallers(PopConfigFunc)) {
    Function *TheFunc = PopCall->getFunction();
    auto Allocas = FuncAllocas.lookup(TheFunc);
    if (Allocas.empty()) {
      continue;
    }

    // ptr nonnull %grid_dim.i, ptr nonnull %block_dim.i, ptr nonnull %shmem_size.i, ptr nonnull %stream.i
    IRBuilder<> IRB(PopCall);
    IRB.CreateStore(IRB.CreateLoad(IRB.getInt64Ty(), Allocas[0]), PopCall->getArgOperand(0));
    IRB.CreateStore(IRB.CreateLoad(IRB.getInt32Ty(), Allocas[1]), IRB.CreateConstInBoundsGEP1_64(IRB.getInt8Ty(), PopCall->getArgOperand(0), 8));
    
    IRB.CreateStore(IRB.CreateLoad(IRB.getInt64Ty(), Allocas[2]), PopCall->getArgOperand(1));
    IRB.CreateStore(IRB.CreateLoad(IRB.getInt32Ty(), Allocas[3]), IRB.CreateConstInBoundsGEP1_64(IRB.getInt8Ty(), PopCall->getArgOperand(1), 8));
    
    IRB.CreateStore(IRB.CreateLoad(IRB.getInt64Ty(), Allocas[4]), PopCall->getArgOperand(2));
    IRB.CreateStore(IRB.CreateLoad(IRB.getPtrTy(), Allocas[5]), PopCall->getArgOperand(3));

    PopCall->replaceAllUsesWith(Constant::getNullValue(PopCall->getType()));
    PopCall->eraseFromParent();
  }
  }
}

class ReactantBase {
public:
  std::vector<std::string> gpubins;
  std::string outfile;
  ReactantBase(const std::vector<std::string> &gpubins, std::string outfile)
      : gpubins(gpubins), outfile(outfile) {}

  bool run(Module &M) {
    bool changed = true;

    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << " pre fix: " << M << "\n";
    fixup(M);
    auto discard = M.getContext().shouldDiscardValueNames();
    M.getContext().setDiscardValueNames(false);
    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << " post fix: " << M << "\n";

    for (auto bin : gpubins) {
      SMDiagnostic Err;
      auto mod2 = llvm::parseIRFile(bin + ".re_export", Err, M.getContext());
      if (!mod2) {
        Err.print(/*ProgName=*/"LLVMToMLIR", llvm::errs());
        exit(1);
      }

      for (std::string T : {"", "f"}) {
        for (std::string name :
             {"sin",       "cos",     "tan",        "log2",    "exp",
              "exp2",      "exp10",   "cosh",       "sinh",    "tanh",
              "atan2",     "atan",    "asin",       "acos",    "log",
              "log10",     "log1p",   "acosh",      "asinh",   "atanh",
              "expm1",     "hypot",   "rhypot",     "norm3d",  "rnorm3d",
              "norm4d",    "rnorm4d", "norm",       "rnorm",   "cbrt",
              "rcbrt",     "j0",      "j1",         "y0",      "y1",
              "yn",        "jn",      "erf",        "erfinv",  "erfc",
              "erfcx",     "erfcinv", "normcdfinv", "normcdf", "lgamma",
              "ldexp",     "scalbn",  "frexp",      "modf",    "fmod",
              "remainder", "remquo",  "powi",       "tgamma",  "round",
              "fdim",      "ilogb",   "logb",       "isinf",   "pow",
              "sqrt",      "finite",  "fabs",       "fmax"}) {
          std::string nvname = "__nv_" + name;
          std::string llname = "llvm." + name + ".";
          std::string mathname = name;

          if (T == "f") {
            mathname += "f";
            nvname += "f";
            llname += "f32";
          } else {
            llname += "f64";
          }

          if (auto F = mod2->getFunction(llname)) {
            F->deleteBody();
          }
        }
      }
      {

        PassBuilder PB;
        LoopAnalysisManager LAM;
        FunctionAnalysisManager FAM;
        CGSCCAnalysisManager CGAM;
        ModuleAnalysisManager MAM;
        PB.registerModuleAnalyses(MAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        GlobalOptPass().run(*mod2, MAM);
      }
      for (auto &F : *mod2) {
        if (!F.empty())
          F.setLinkage(Function::LinkageTypes::InternalLinkage);
      }
      if (getenv("DEBUG_REACTANT"))
        llvm::errs() << " mod2: " << *mod2 << "\n";

      SmallVector<std::string> toInternalize;
      if (auto RF = M.getFunction("__cudaRegisterFunction")) {
        for (auto U : make_early_inc_range(RF->users())) {
          if (auto CI = dyn_cast<CallBase>(U)) {
            if (CI->getCalledFunction() != RF)
              continue;

            Value *F2 = CI->getArgOperand(1);
            Value *name = CI->getArgOperand(2);
            while (auto CE = dyn_cast<ConstantExpr>(F2)) {
              F2 = CE->getOperand(0);
            }
            while (auto CE = dyn_cast<ConstantExpr>(name)) {
              name = CE->getOperand(0);
            }
            StringRef nameVal;
            if (auto GV = dyn_cast<GlobalVariable>(name))
              if (GV->isConstant())
                if (auto C = GV->getInitializer())
                  if (auto CA = dyn_cast<ConstantDataArray>(C))
                    if (CA->getType()->getElementType()->isIntegerTy(8) &&
                        CA->isCString())
                      nameVal = CA->getAsCString();
            auto F22 = dyn_cast<Function>(F2);
            if (!F22)
              continue;

            if (nameVal.size())
              if (auto MF = mod2->getFunction(nameVal)) {
                MF->setName("reactant$" + F22->getName());
                MF->setCallingConv(llvm::CallingConv::C);
                MF->setLinkage(Function::LinkageTypes::LinkOnceODRLinkage);
                F22->setLinkage(Function::LinkageTypes::InternalLinkage);
                toInternalize.push_back(MF->getName().str());
                CI->eraseFromParent();
              }
          }
        }
      }

      auto handler = M.getContext().getDiagnosticHandler();
      Linker L(M);
      L.linkInModule(std::move(mod2));
      M.getContext().setDiagnosticHandler(std::move(handler));
      for (auto name : toInternalize)
        if (auto F = M.getFunction(name)) {
          F->setLinkage(Function::LinkageTypes::InternalLinkage);
        }
    }

    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << "post link: " << M << "\n";

    if (auto F = M.getFunction("__mlir_cuda_caller_phase2")) {
      for (auto U : make_early_inc_range(F->users())) {
        auto CI = cast<CallInst>(U);
        SmallVector<Value *> args;
        for (auto &arg : CI->args()) {
          args.push_back(arg.get());
        }
        auto F = cast<Function>(args[0]);
        auto ptr = args.pop_back_val();
        IRBuilder<> B(CI);

        auto ptrty = PointerType::getUnqual(F->getContext());
        for (size_t i = 0; i < F->getFunctionType()->getNumParams(); i++) {
          auto gep = B.CreateConstInBoundsGEP1_32(ptrty, ptr, i);
          auto ld = B.CreateLoad(ptrty, gep);
          if (auto T = F->getParamByValType(i)) {
            (void)T;
            args.push_back(ld);
          } else {
            auto ld2 = B.CreateLoad(F->getFunctionType()->getParamType(i), ld);
            args.push_back(ld2);
          }
        }

        auto MlirLaunchFunc = M.getOrInsertFunction(
            "__mlir_cuda_caller_phase3",
            FunctionType::get(Type::getVoidTy(M.getContext()), {}, true));

        B.CreateCall(MlirLaunchFunc, args);
        CI->eraseFromParent();
      }
    }

    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << "post link2: " << M << "\n";

    fixup(M);
    for (auto todel : {"__cuda_register_globals", "__cuda_module_ctor",
                       "__cuda_module_dtor"}) {
      if (auto F = M.getFunction(todel)) {
        F->replaceAllUsesWith(Constant::getNullValue(F->getType()));
        F->eraseFromParent();
      }
    }

    if (auto GV = M.getGlobalVariable("llvm.global_ctors")) {
      ConstantArray *CA = dyn_cast<ConstantArray>(GV->getInitializer());
      if (CA) {

        bool changed = false;
        SmallVector<Constant *> newOperands;
        for (Use &OP : CA->operands()) {
          if (isa<ConstantAggregateZero>(OP)) {
            changed = true;
            continue;
          }
          ConstantStruct *CS = cast<ConstantStruct>(OP);
          if (isa<ConstantPointerNull>(CS->getOperand(1))) {
            changed = true;
            continue;
          }
          newOperands.push_back(CS);
        }
        if (changed) {
          if (newOperands.size() == 0) {
            GV->eraseFromParent();
          } else {
            auto EltTy = newOperands[0]->getType();
            ArrayType *NewType = ArrayType::get(EltTy, newOperands.size());
            auto CT = ConstantArray::get(NewType, newOperands);

            // Create the new global variable.
            GlobalVariable *NG = new GlobalVariable(
                M, NewType, GV->isConstant(), GV->getLinkage(),
                /*init*/ CT, /*name*/ "", GV, GV->getThreadLocalMode(),
                GV->getAddressSpace());

            NG->copyAttributesFrom(GV);
            NG->takeName(GV);
            GV->replaceAllUsesWith(NG);
            GV->eraseFromParent();
          }
        }
      }
    }

    {
      PassBuilder PB;
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;
      PB.registerModuleAnalyses(MAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      GlobalOptPass().run(M, MAM);
    }

    auto lib = dlopen(Passes.c_str(), RTLD_LAZY | RTLD_DEEPBIND);
    if (!lib) {
      llvm::errs() << " could not open " << Passes.c_str() << " - " << dlerror()
                   << "\n";
    }
    auto sym = dlsym(lib, "runLLVMToMLIRRoundTrip");
    if (!sym) {
      llvm::errs() << " could not find sym\n";
    }
    auto runLLVMToMLIRRoundTrip =
        (std::string(*)(std::string, std::string, std::string, std::string))sym;
    if (runLLVMToMLIRRoundTrip) {
      std::string MStr;
      llvm::raw_string_ostream ss(MStr);
      ss << M;
      auto newMod =
          runLLVMToMLIRRoundTrip(MStr, outfile, ReactantBackend.getValue(),
                                 DeviceLibraries.getValue());
      if (newMod.empty()) {
        M.getContext().diagnose(DiagnosticInfoUnsupported(
            *M.begin(), "Reactant: failed to run mlir passes"));
        return changed;
      }
      M.dropAllReferences();

      M.getGlobalList().clear();
      M.getFunctionList().clear();
      M.getAliasList().clear();
      M.getIFuncList().clear();
      M.getComdatSymbolTable().clear();

      llvm::SMDiagnostic Err;
      auto llvmModule = llvm::parseIR(
          llvm::MemoryBufferRef(newMod, "conversion"), Err, M.getContext());

      if (!llvmModule) {
        llvm::errs() << " newMod: " << newMod << "\n";
        Err.print(/*ProgName=*/"LLVMToMLIR", llvm::errs());
        exit(1);
      }
      auto handler = M.getContext().getDiagnosticHandler();
      Linker L(M);
      L.linkInModule(std::move(llvmModule), Linker::Flags::OverrideFromSrc);
      M.getContext().setDiagnosticHandler(std::move(handler));
    }
    M.getContext().setDiscardValueNames(discard);

    return changed;
  }
};

} // namespace

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/Passes/PassPlugin.h"

class ReactantNewPM final : public ReactantBase,
                            public AnalysisInfoMixin<ReactantNewPM> {
  friend struct llvm::AnalysisInfoMixin<ReactantNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  ReactantNewPM(const std::vector<std::string> &gpubins, std::string outfile)
      : ReactantBase(gpubins, outfile) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    return ReactantBase::run(M) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

class ExporterNewPM final : public AnalysisInfoMixin<ExporterNewPM> {
  friend struct llvm::AnalysisInfoMixin<ExporterNewPM>;

private:
  static llvm::AnalysisKey Key;

  // Add bodies returning the poison value to declarations of Enzyme internal
  // functions. These are not supposed to be called.
  void definePoisonEnzymeCalls(llvm::Module &M) {
    for (llvm::Function &Fn : M) {
      if (!Fn.isDeclaration())
        continue;
      if (!(Fn.getName().contains("__enzyme_float") ||
            Fn.getName().contains("__enzyme_double") ||
            Fn.getName().contains("__enzyme_integer") ||
            Fn.getName().contains("__enzyme_pointer") ||
            Fn.getName().contains("__enzyme_virtualreverse") ||
            Fn.getName().contains("__enzyme_call_inactive") ||
            Fn.getName().contains("__enzyme_autodiff") ||
            Fn.getName().contains("__enzyme_fwddiff") ||
            Fn.getName().contains("__enzyme_fwdsplit") ||
            Fn.getName().contains("__enzyme_augmentfwd") ||
            Fn.getName().contains("__enzyme_augmentsize") ||
            Fn.getName().contains("__enzyme_reverse") ||
            Fn.getName().contains("__enzyme_truncate") ||
            Fn.getName().contains("__enzyme_batch") ||
            Fn.getName().contains("__enzyme_error_estimate") ||
            Fn.getName().contains("__enzyme_trace") ||
            Fn.getName().contains("__enzyme_condition") ||
            Fn.getName().contains("__enzyme_set_checkpointing") ||
            Fn.getName().contains("__enzyme_set_mincut")))
        continue;

      auto *BB = llvm::BasicBlock::Create(M.getContext(), "", &Fn);
      llvm::IRBuilder<> Builder(BB);

      llvm::Type *RT = Fn.getReturnType();
      if (RT->isVoidTy()) {
        Builder.CreateRetVoid();
        continue;
      }

      auto *Poison = llvm::PoisonValue::get(RT);
      Builder.CreateRet(Poison);
    }
  }

public:
  using Result = llvm::PreservedAnalyses;
  std::string firstfile;
  ExporterNewPM(std::string file) : firstfile(file) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    std::string filename = firstfile + ".re_export";

    std::error_code EC;
    llvm::raw_fd_ostream file(filename, EC); //, llvm::sys::fs::OF_Text);

    if (EC) {
      llvm::errs() << "Error opening file: " << EC.message() << "\n";
      exit(1);
    }

    file << M;
    definePoisonEnzymeCalls(M);
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

#undef DEBUG_TYPE
AnalysisKey ReactantNewPM::Key;
AnalysisKey ExporterNewPM::Key;

#include "llvm/Passes/PassBuilder.h"

extern "C" void registerExporter(llvm::PassBuilder &PB, std::string file) {

  auto loadNVVM = [](ModulePassManager &MPM, OptimizationLevel) {
    MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true));
  };

  // We should register at vectorizer start for consistency, however,
  // that requires a functionpass, and we have a modulepass.
  // PB.registerVectorizerStartEPCallback(loadPass);
  PB.registerPipelineStartEPCallback(loadNVVM);
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadNVVM);

#if LLVM_VERSION_MAJOR >= 20
  auto loadPass =
      [=](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase)
#else
  auto loadPass = [=](ModulePassManager &MPM, OptimizationLevel Level)
#endif
  { MPM.addPass(ExporterNewPM(file)); };

  // TODO need for perf reasons to move Enzyme pass to the pre vectorization.
  PB.registerOptimizerEarlyEPCallback(loadPass);

  auto loadLTO = [loadPass](ModulePassManager &MPM, OptimizationLevel Level) {
#if LLVM_VERSION_MAJOR >= 20
    loadPass(MPM, Level, ThinOrFullLTOPhase::None);
#else
    loadPass(MPM, Level);
#endif
  };
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadLTO);
}

extern "C" void registerReactantAndPassPipeline(llvm::PassBuilder &PB,
                                                bool augment = false) {}

extern "C" void registerReactant(llvm::PassBuilder &PB,
                                 std::vector<std::string> gpubinaries,
                                 std::string outfile) {

  llvm::errs() << " registering reactant\n";
#if LLVM_VERSION_MAJOR >= 20
  auto loadPass =
      [=](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase)
#else
  auto loadPass = [=](ModulePassManager &MPM, OptimizationLevel Level)
#endif
  { MPM.addPass(ReactantNewPM(gpubinaries, outfile)); };

  PB.registerPipelineParsingCallback(
      [=](llvm::StringRef Name, llvm::ModulePassManager &MPM,
          llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "reactant") {
          MPM.addPass(ReactantNewPM(gpubinaries, outfile));
          return true;
        }
        return false;
      });

  // TODO need for perf reasons to move Enzyme pass to the pre vectorization.
  PB.registerOptimizerEarlyEPCallback(loadPass);

  auto loadLTO = [loadPass](ModulePassManager &MPM, OptimizationLevel Level) {
#if LLVM_VERSION_MAJOR >= 20
    loadPass(MPM, Level, ThinOrFullLTOPhase::None);
#else
    loadPass(MPM, Level);
#endif
  };
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadLTO);
}

extern "C" void registerReactant2(llvm::PassBuilder &PB) {
  registerReactant(PB, {}, "");
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ReactantNewPM", "v0.1", registerReactant2};
}
