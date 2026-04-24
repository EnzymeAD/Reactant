#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

static cl::opt<std::string> InputFile(cl::Positional, cl::desc("<input bitcode>"), cl::Required);

extern "C" void registerReactant(PassBuilder &PB, std::vector<std::string> gpubins, std::string outfile);

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    cl::ParseCommandLineOptions(argc, argv, "reactant-opt\n");

    LLVMContext Ctx;
    Ctx.setDiscardValueNames(false);
    SMDiagnostic Err;
    auto M = parseIRFile(InputFile, Err, Ctx);
    if (!M) {
        Err.print(argv[0], errs());
        return 1;
    }

    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    registerReactant(PB, {}, "");

    PB.registerModuleAnalyses(MAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(PassManager<Function>()));

    // parse and run the pipeline
    if (auto Err = PB.parsePassPipeline(MPM, "reactant")) {
        errs() << "Error: " << toString(std::move(Err)) << "\n";
        return 1;
    }

    MPM.run(*M, MAM);
    return 0;
}