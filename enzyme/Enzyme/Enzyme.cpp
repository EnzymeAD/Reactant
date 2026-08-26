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
#include <llvm/IR/DerivedTypes.h>
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
#include "llvm/Transforms/Utils/Local.h"
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
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/IOSandbox.h"
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

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"


#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Analysis/PostDominators.h"

#include "Enzyme/PassUtils.h"
#include "Enzyme/PreserveNVVM.h"
#include "Enzyme/Utils.h"

#ifndef REACTANT_USE_LINKED_RAISE
#define REACTANT_USE_LINKED_RAISE 0
#endif

#if REACTANT_USE_LINKED_RAISE
#include "src/enzyme_ad/jax/raise.h"
#else
struct MLIRRoundTripOptions {
  bool dataflow;
  bool markReadonly;
  bool preADLowerAffine;
  bool splitMultiResults;
  bool removeAtomics;
  bool sortBlockMemory;
  bool hoistLoopAllocations;
  bool lowerInvoke;
  bool verifyEach;
  bool parallelCanonicalize;
  int lateSink;
};
#endif

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

llvm::cl::opt<bool>
    DataFlowActivity("reactant-dataflow", cl::init(false), cl::Hidden,
                     cl::desc("Use data-flow activity analysis"));

llvm::cl::opt<bool>
    MarkReadOnly("reactant-mark-readonly", cl::init(false), cl::Hidden,
                 cl::desc("If using data-flow analysis, mark pointers that are only read from"));

llvm::cl::opt<bool>
    SplitMultiResults("reactant-split-multi-results", cl::init(false), cl::Hidden,
                 cl::desc("Split certain operations that produce multiple results"));

llvm::cl::opt<bool>
    PreADLowerAffine("reactant-pre-ad-lower-affine", cl::init(false), cl::Hidden,
                 cl::desc("Lower affine dialect operations right before differentiation"));

llvm::cl::opt<bool>
    RemoveAtomics("reactant-remove-atomics", cl::init(false), cl::Hidden,
                 cl::desc("Replace provably race-free atomics with plain load/store after differentiation"));

llvm::cl::opt<bool>
    SortBlockMemory("reactant-sort-block-memory", cl::init(false), cl::Hidden,
                 cl::desc("Hoist non-overlapping loads to the start of their block and sink stores to the end"));

llvm::cl::opt<bool>
    HoistLoopAllocations("reactant-hoist-loop-allocations", cl::init(true), cl::Hidden,
                 cl::desc("Hoist allocations out of loops after differentiation"));

llvm::cl::opt<bool>
    LowerInvoke("reactant-lower-invoke", cl::init(true), cl::Hidden,
                 cl::desc("Lower invoke to call before import, discarding exception handling; "
                          "0 keeps unwind edges, leaving throwing or catching functions in cf form"));

llvm::cl::opt<bool> VerifyEach(
    "reactant-verify-each", cl::init(false), cl::Hidden,
    cl::desc("Verify the module after every pass of the raising pipeline "
             "instead of once after it; per-pass verification is a third "
             "of the pipeline on large translation units"));

llvm::cl::opt<bool> ParallelCanonicalize(
    "reactant-parallel-canonicalize", cl::init(false), cl::Hidden,
    cl::desc("Let the raising pipeline's canonicalize-parallel use the "
             "context's thread pool; off by default so every compile job "
             "of a parallel build does not spawn its own pool"));

llvm::cl::opt<int> LateSink(
    "reactant-late-sink", cl::init(2), cl::Hidden,
    cl::desc("Mode for the GPU serializer's late sink of cheap address "
             "computations: 0 disables it, 1 sinks only within the "
             "defining loop, 2 also rematerializes into deeper loops"));

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

  // Map of runtime function, index of the entry fn
  std::pair<const char *, int> runtime_fns[] = {
      {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", 1},
      {"cudaFuncGetAttributes", 1},
      {"cudaFuncGetName", 1},
      {"cudaFuncSetAttribute", 0},
      {"cudaFuncSetCacheConfig", 0},
      {"cudaLaunchKernelExC", 1},
  };

  auto LaunchKernelFunc = M.getFunction(cudaLaunchSymbolName);
  bool AnyRuntimeUse = LaunchKernelFunc != nullptr;
  for (auto &pair : runtime_fns)
    if (M.getFunction(pair.first))
      AnyRuntimeUse = true;
  if (!AnyRuntimeUse)
    return;

  SmallPtrSet<Function *, 8> HostStubFuncs;

  // For all calls to cudaLaunchKernel, find the stub function, and replace
  // the stub function with a empty declaration of reactant$stubname, which will now
  // correspond to the actual device code. This is because clang for cuda will emit two unrelated functions
  // with the same name, one on device (containing device code) and one on host, containing cudaPop and the call to
  // the kernel launch. As we do not want to confuse one and the other, we explicitly separate them.
  // Furthermore, we replace cudaLaunchKernel with a call to __mlir_cuda_caller_phase2 and extract
  // the corresponding launch sizes (block/thread dim) as separate arguments, as parsed from cudaLaunchKernel.
  // The list of host-side stub functions used here are placed within `HostStubFuncs`.
  // Example code before:
  //  void stubFunc() {
  //     cudaPop();
  //     cudaLaunch(@stubFunc, x | y, z, ...);
  //  }
  //  void main() {
  //     cudaPush(x | y, z, ...);
  //     stubFunc();
  // }
  // 
  // Example code before:
  //  void stubFunc() {
  //     cudaPop();
  //     __mlir_cuda_caller_phase2(@reactant$stubFunc, x, y, z, ...);
  //  }
  //  void main() {
  //     cudaPush(x | y, z, ...);
  //     stubFunc();
  // }
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

    // This is a declaration of the new device-function we will use. It is empty because it will be replaced by the
    // device function of the corresponding name, during subsequent linking
    auto NewFn = M.getOrInsertFunction(("reactant$" + StubFunc->getName()).str(), StubFunc->getFunctionType(), StubFunc->getAttributes());
    if (auto *NF = dyn_cast<Function>(NewFn.getCallee()))
      NF->addFnAttr("polygeist.host_symbol", StubFunc->getName());
    
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
    // The stub inliner otherwise rewrites this call into an invoke when the
    // stub was invoked inside an exception scope; the launcher never unwinds.
    if (auto *LF = dyn_cast<Function>(MlirLaunchFunc.getCallee()))
      LF->addFnAttr(Attribute::NoUnwind);

    Builder.CreateCall(MlirLaunchFunc, Args);
    HostStubFuncs.insert(StubFunc);
    if (auto II = dyn_cast<InvokeInst>(CI)) {
      Builder.CreateBr(II->getNormalDest());
      II->getUnwindDest()->removePredecessor(II->getParent());
    }
    CI->eraseFromParent();
  }

  // For all callers of the host-side stub function, attempt to inline where possible.
  // 
  // Example code before:
  //  void stubFunc() {
  //     cudaPop();
  //     __mlir_cuda_caller_phase2(@reactant$stubFunc, x, y, z, ...);
  //  }
  //  void main() {
  //     cudaPush(x | y, z, ...);
  //     stubFunc();
  // }
  //
  // Example code after:
  //  void stubFunc() {
  //     cudaPop();
  //     __mlir_cuda_caller_phase2(@reactant$stubFunc, x, y, z, ...);
  //  }
  //  void main() {
  //     cudaPush(x | y, z, ...);
  //     cudaPop();
  //     __mlir_cuda_caller_phase2(@reactant$stubFunc, x, y, z, ...);
  // }
  //
  // Note that stubFunc is still left around. Also note that, inlining may not succeed, if,
  // for example, stubFunc is captured and called indirectly, like as follows
  //
  // Example indirect code before:
  //  void stubFunc() {
  //     cudaPop();
  //     __mlir_cuda_caller_phase2(@reactant$stubFunc, x, y, z, ...);
  //  }
  //  void indirect(void* tocall) {
  //     cudaPush(x | y, z, ...);
  //     tocall();
  //  }
  //  void main() {
  //     indirect(stubFunc);
  // }
  for (auto StubFunc : HostStubFuncs) {
    for (CallBase *CI : gatherCallers(StubFunc)) {
       InlineFunctionInfo IFI;
          InlineResult Res =
              InlineFunction(*CI, IFI, /*MergeAttributes=*/false);
          assert(Res.isSuccess());
    }
  }

  // For all callers of the host-side stub function, that really should have
  // called the device version of the stub function, replace known cuda runtime
  // calls to the host stub function with the new device stub function.
  // 
  // Currently we just replace uses known from the runtime_fns map above. This
  // is not exhaustive, but required for code calling it.
  //
  // Example code before:
  //  void stubFunc() {
  //     cudaPop();
  //     __mlir_cuda_caller_phase2(@reactant$stubFunc, x, y, z, ...);
  //  }
  //  void main() {
  //     cudaFuncSetCacheConfig(@stubFunc, ...);
  // }
  //
  // Example code after:
  //  void stubFunc() {
  //     cudaPop();
  //     __mlir_cuda_caller_phase2(@reactant$stubFunc, x, y, z, ...);
  //  }
  //  void main() {
  //     cudaFuncSetCacheConfig(@reactant$stubFunc, ...);
  //  }
  IRBuilder<> Builder(M.getContext());
  const char *GetDeviceFromHostFuncName = "__reactant$get_device_from_host";
  FunctionCallee GetDeviceFromHost = M.getOrInsertFunction(
      GetDeviceFromHostFuncName,
      FunctionType::get(Builder.getPtrTy(), {Builder.getPtrTy()},
                        /*isVarArg=*/false));
  for (auto &pair : runtime_fns) {
    if (auto rtFunc = M.getFunction(pair.first)) {
      for (CallBase *CI : gatherCallers(rtFunc)) {
        auto StubFunc = dyn_cast<Function>(CI->getArgOperand(pair.second));
        if (!StubFunc) {
          if (auto Call = dyn_cast<CallInst>(CI->getArgOperand(pair.second));
              Call && Call->getCalledFunction() &&
              Call->getCalledFunction()->getName() == GetDeviceFromHostFuncName)
            continue;
          Builder.SetInsertPoint(CI);
          CI->setArgOperand(pair.second,
                            Builder.CreateCall(GetDeviceFromHost,
                                               CI->getArgOperand(pair.second)));
          continue;
        }
        if (StubFunc->getName().starts_with("reactant$"))
          continue;
        auto NewFn = M.getOrInsertFunction(
            ("reactant$" + StubFunc->getName()).str(),
            StubFunc->getFunctionType(), StubFunc->getAttributes());
        if (auto *NF = dyn_cast<Function>(NewFn.getCallee()))
          NF->addFnAttr("polygeist.host_symbol", StubFunc->getName());
        CI->setArgOperand(pair.second, NewFn.getCallee());
      }
    }
  }

  // We now retain cudaPush and cudaPop. These functions no longer have any meaning
  // for the actual kernel call itself, since we have already parsed the cuda launch boundaries
  // from cudaLaunchFunc earlier. However, these functions still represent a stack of data transfers.
  // We do an extremely primitive form of mem2reg for the push/pop so we can actually see through
  // the invocation and figure out which arguments map where.

  // The push/pop pairing below requires the pop to post-dominate the push,
  // and in exception-aware code both come in as invokes whose unwind edge
  // escapes to a landing pad, which defeats that. Neither function can
  // throw, so lower their invokes to calls first.
  for (auto name : {cudaPushConfigName, cudaPopConfigName})
    if (Function *F = M.getFunction(name)) {
      F->setDoesNotThrow();
      for (CallBase *CB : gatherCallers(F))
        if (auto *II = dyn_cast<InvokeInst>(CB))
          changeToCall(II);
    }

  DenseMap<Function *, SetVector<CallBase*>> Pushes;
  if (auto PushConfigFunc = M.getFunction(cudaPushConfigName)) {
    for (CallBase *CI : gatherCallers(PushConfigFunc)) {
      Function *TheFunc = CI->getFunction();
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
      Pushes[TheFunc].insert(CI);
    }
  }
    
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


  for (auto &pair : Pushes) {
    auto NewF = pair.first;
	  {
    auto PA = InstCombinePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }

  {
    SimplifyCFGOptions scfgo;
    auto PA = SimplifyCFGPass(scfgo).run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }
  }

  Pushes.clear();

  // Per each function, this alloca list repreesents the corrseponding args to cuda-mem2reg within the function boundary.
  DenseMap<Function *, SmallVector<AllocaInst *, 6>> FuncAllocas;

  // Find all calls to push, add to the alloca, and remove the original push.
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
      Pushes[TheFunc].insert(CI);
    }
  }

  // Find all calls to pop, lookup the alloca (if available), and remove the original pop.
  if (  auto PopConfigFunc = M.getFunction(cudaPopConfigName)) {

    DenseMap<Function *, SetVector<CallBase*>> Pops;

    for (CallBase *PopCall : gatherCallers(PopConfigFunc)) {
      Function *TheFunc = PopCall->getFunction();
      Pops[TheFunc].insert(PopCall);
    }

    SmallVector<Function*, 1> todo;
    for (auto &pair : Pops) {
      if (Pushes.contains(pair.first)) {
        todo.push_back(pair.first);
      }
    }

    while (!todo.empty()) {
      auto TheFunc = todo.pop_back_val();
      const auto &PopList = Pops.lookup(TheFunc);
      for (auto PopCall : PopList) {
        auto Allocas = FuncAllocas.lookup(TheFunc);
        if (Allocas.empty()) {
          continue;
        }

        llvm::DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*TheFunc);
        llvm::PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(*TheFunc);

        // find corresponding pop for push
        CallBase* PushCall = nullptr;
        const auto &PushList = Pushes.lookup(TheFunc);
        for (auto push : PushList) {
          // Check if push dominates the pop, and the pop dominates the push, if so we delete
          // If there is an earlier push we found that has this property, pick the latter of the two.

          if (!DT.dominates(push, PopCall)) {
            continue;
          }

          if (!PDT.dominates(PopCall, push)) {
            continue;
          }

          if (!PushCall) {
            PushCall = push;
          } else {
            if (DT.dominates(PushCall, push)) {
              PushCall = push;
            } else if (DT.dominates(push, PushCall)) {
              continue;
            } else {
              PushCall = nullptr;
              break;
            }
          }
        }

        if (!PushCall) {
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

        if (PushList.size() > 1 && PopList.size() > 1) {
          todo.push_back(TheFunc);
        }
        
	Pushes[TheFunc].remove(PushCall);
	Pops[TheFunc].remove(PopCall);
        
        PopCall->eraseFromParent();
        PushCall->eraseFromParent();
	break;
      }
    }
  }
}

static std::string reExportPath(llvm::StringRef src, bool besideSource) {
  if (besideSource)
    return (src + ".re_export").str();
  llvm::SmallString<128> tmp;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, tmp);
  llvm::sys::path::append(
      tmp, "reactant-" + llvm::utohexstr(llvm::hash_value(src)) + ".re_export");
  return std::string(tmp);
}

class ReactantBase {
public:
  std::vector<std::string> gpubins;
  std::string outfile;
  ReactantBase(const std::vector<std::string> &gpubins, std::string outfile)
      : gpubins(gpubins), outfile(outfile) {}

  bool run(Module &M) {
    auto DisableSandbox = llvm::sys::sandbox::scopedDisable();
    bool changed = true;

    // Textual IR cannot be read into a context that drops value names, which
    // is what a -O2 compile hands us.
    auto discard = M.getContext().shouldDiscardValueNames();
    M.getContext().setDiscardValueNames(false);

    SmallVector<std::unique_ptr<Module>> deviceMods;
    for (auto bin : gpubins) {
      SMDiagnostic Err;
      auto path = reExportPath(bin, /*besideSource=*/true);
      auto mod2 = llvm::parseIRFile(path, Err, M.getContext());
      if (!mod2) {
        path = reExportPath(bin, /*besideSource=*/false);
        mod2 = llvm::parseIRFile(path, Err, M.getContext());
      }
      if (getenv("DEBUG_REACTANT"))
        llvm::errs() << " device module " << path << ": "
                     << (mod2 ? (mod2->empty() ? "empty" : "present")
                              : Err.getMessage())
                     << "\n";
      deviceMods.emplace_back(std::move(mod2));
    }

    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << " pre fix: " << M << "\n";
    fixup(M);
    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << " post fix: " << M << "\n";

    for (auto &mod2 : deviceMods) {
      if (!mod2) {
        llvm::errs() << "LLVMToMLIR: could not read the device module\n";
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
      // The device module carries its own copies of globals the host also
      // defines -- statics in headers compiled for both sides. They are
      // separate objects: the host keeps its copy, and the device copy is
      // internalized so linking cannot see two strong definitions of one
      // name. Special globals (llvm.used and friends, appending linkage)
      // keep their semantics.
      for (auto &G : mod2->globals()) {
        if (!G.hasInitializer())
          continue;
        if (G.hasAppendingLinkage() || G.getName().starts_with("llvm."))
          continue;
        // Enzyme's activity markers are recognized by name; a conflict
        // rename would turn enzyme_dup into enzyme_dup.2 and lose them.
        if (G.getName().contains("enzyme_"))
          continue;
        G.setComdat(nullptr);
        G.setLinkage(GlobalValue::InternalLinkage);
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
                // A registered kernel whose device body is not in this
                // module stays a declaration, and a declaration cannot
                // wear link-once linkage.
                if (MF->isDeclaration())
                  continue;
                MF->setName("reactant$" + F22->getName());
                // Record which host symbol this kernel was registered for, so
                // later stages need not infer it from the name.
                MF->addFnAttr("polygeist.host_symbol", F22->getName());
                MF->setCallingConv(llvm::CallingConv::C);
                MF->setLinkage(Function::LinkageTypes::LinkOnceODRLinkage);
                // The stub keeps its comdat only as long as it keeps external
                // linkage. Made internal inside one, the linker drops the
                // section when another object wins the group and the calls
                // this unit still makes to it have nowhere to go: "defined in
                // discarded section".
                F22->setComdat(nullptr);
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
          F->setComdat(nullptr);
          F->setLinkage(Function::LinkageTypes::InternalLinkage);
        }
    }

    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << "post link: " << M << "\n";

    if (auto F = M.getFunction("__mlir_cuda_caller_phase2")) {
      for (auto U : make_early_inc_range(F->users())) {
        // The stub inliner rewrites the phase2 call into an invoke when the
        // stub was invoked inside an exception scope (belt-and-braces: the
        // nounwind attribute at creation should prevent this).
        auto CI = cast<CallBase>(U);
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
        if (auto II = dyn_cast<InvokeInst>(CI)) {
          B.CreateBr(II->getNormalDest());
          II->getUnwindDest()->removePredecessor(II->getParent());
        }
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

    // Hand the module over as bitcode. It crosses this boundary as bytes either
    // way, and bitcode is the cheaper and more faithful spelling of it: on
    // MFEM's dFEM tests it is a fifth the size of the textual form and parses
    // in half the time, and it does not depend on a printer and a parser
    // agreeing about syntax. The reader on the far side is llvm::parseIR, which
    // sniffs the bitcode magic and dispatches, so it takes either.
    std::string MStr;
    llvm::raw_string_ostream ss(MStr);
    llvm::WriteBitcodeToFile(M, ss);
    // The flag wins when given; otherwise REACTANT_LOWER_INVOKE=0/1 decides,
    // so a deployment can flip it without threading -mllvm flags through its
    // build system.
    bool lowerInvoke = LowerInvoke.getValue();
    if (LowerInvoke.getNumOccurrences() == 0)
      if (const char *env = getenv("REACTANT_LOWER_INVOKE"))
        lowerInvoke = env[0] && env[0] != '0';
    bool verifyEach = VerifyEach.getValue();
    if (VerifyEach.getNumOccurrences() == 0)
      if (const char *env = getenv("REACTANT_VERIFY_EACH"))
        verifyEach = env[0] && env[0] != '0';
    bool parallelCanonicalize = ParallelCanonicalize.getValue();
    if (ParallelCanonicalize.getNumOccurrences() == 0)
      if (const char *env = getenv("REACTANT_CANON_PARALLEL"))
        parallelCanonicalize = env[0] && env[0] != '0';
    int lateSink = LateSink.getValue();
    if (LateSink.getNumOccurrences() == 0)
      if (const char *env = getenv("REACTANT_LATE_SINK"))
        lateSink = atoi(env);
    MLIRRoundTripOptions options{
        .dataflow = DataFlowActivity.getValue(),
        .markReadonly = MarkReadOnly.getValue(),
        .preADLowerAffine = PreADLowerAffine.getValue(),
        .splitMultiResults = SplitMultiResults.getValue(),
        .removeAtomics = RemoveAtomics.getValue(),
        .sortBlockMemory = SortBlockMemory.getValue(),
        .hoistLoopAllocations = HoistLoopAllocations.getValue(),
        .lowerInvoke = lowerInvoke,
        .verifyEach = verifyEach,
        .parallelCanonicalize = parallelCanonicalize,
        .lateSink = lateSink,
    };

#if REACTANT_USE_LINKED_RAISE 
    auto newMod =
        runLLVMToMLIRRoundTrip(MStr, outfile, ReactantBackend.getValue(),
                               DeviceLibraries.getValue(), &options);
#else
    std::string newMod;

    {
    auto DisableSandbox = llvm::sys::sandbox::scopedDisable();

    // load symbol and run pass
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
        (std::string(*)(std::string, std::string, std::string, std::string, MLIRRoundTripOptions *))sym;
    newMod =
        runLLVMToMLIRRoundTrip(MStr, outfile, ReactantBackend.getValue(),
                               DeviceLibraries.getValue(), &options);
    }
#endif

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
    // The round trip preserved the module asm (a stub translation unit is
    // nothing but its fatbin asm), and the linker below appends the source
    // module's asm to whatever remains here: keeping ours would define
    // fatbinData twice.
    M.removeModuleInlineAsm();

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
    M.getContext().setDiscardValueNames(discard);

    return changed;
  }
};

} // namespace

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/Plugins/PassPlugin.h"

class ReactantNewPM final : public ReactantBase,
                            public PassParent<ReactantNewPM> {
  friend PassParent<ReactantNewPM>;

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

class ExporterNewPM final : public PassParent<ExporterNewPM> {
  friend PassParent<ExporterNewPM>;

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
    auto DisableSandbox = llvm::sys::sandbox::scopedDisable();

    std::error_code EC;
    auto file = std::make_unique<llvm::raw_fd_ostream>(
        reExportPath(firstfile, /*besideSource=*/true), EC);

    if (EC) {
      // Not every source sits somewhere writable: cmake compiles the cuda
      // toolkit's own link.stub, out of the toolkit's directory, for the
      // device link step. Fall back to a temp file the host side names the
      // same way.
      file = std::make_unique<llvm::raw_fd_ostream>(
          reExportPath(firstfile, /*besideSource=*/false), EC);
      if (EC) {
        llvm::errs() << "Error opening file: " << EC.message() << "\n";
        exit(1);
      }
    }

    *file << M;
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

  // The raising pipeline has no custom derivative rules, so nothing here will
  // ever resolve one. Consuming the registration globals still matters -- left
  // standing they are a definition of the same name in every unit that saw the
  // declaration -- but holding the functions they name external and uninlined
  // only keeps dead code, and the kernels it launches, alive.
  auto loadNVVM = [](ModulePassManager &MPM, OptimizationLevel) {
    MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true,
                                  /*PreserveCustomRuleLinkage*/ false));
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

  // The clang plugin turns an enzyme_inactive attribute into a global naming
  // the function it was written on, one per translation unit that saw the
  // declaration. PreserveNVVM is what reads those, moves what they say onto
  // the function, and takes them away again -- left standing they are a
  // definition of the same name in every such unit, which is a link MFEM does
  // not get to the end of. registerExporter below asks for it in the same way.
  // It has no custom derivative rules either, so nothing here will ever
  // resolve one; holding the functions a rule names external and uninlined
  // would only keep dead code, and the kernels it launches, alive.
  auto loadNVVM = [](ModulePassManager &MPM, OptimizationLevel) {
    MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true,
                                  /*PreserveCustomRuleLinkage*/ false));
  };
  PB.registerPipelineStartEPCallback(loadNVVM);
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadNVVM);

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
