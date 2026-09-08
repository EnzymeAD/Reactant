//===- ropt.cpp - The Reactant modular optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `opt` with the Reactant pass pre-registered, so `-passes=reactant` works
// without -load-pass-plugin. Everything else -- reading stdin or a file,
// -S/-o, -O<n>, remarks, --time-passes, --print-after-all -- is LLVM's own
// opt driver, reused rather than reimplemented. Mirrors Enzyme's enzyme-opt.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"

#include <functional>
#include <string>
#include <vector>

using namespace llvm;

// The two arguments the clang plugin supplies to registerReactant, exposed so
// that a standalone run can reach the same behaviour. Both default to what
// registerReactant2 hardcodes, so plain `reactant-opt -passes=reactant` is
// unchanged.
static cl::list<std::string> GPUBinaries(
    "reactant-gpu-binary",
    cl::desc("Source file whose device-side module the Reactant pass should "
             "merge into the host module; the pass reads it back from "
             "<path>.re_export, written by an earlier device compile. This is "
             "registerReactant's `gpubinaries` argument -- despite the name it "
             "takes source paths, not binaries. Repeatable."),
    cl::value_desc("path"));

static cl::opt<std::string> ReactantOutfile(
    "reactant-outfile",
    cl::desc("Base path for the Reactant pass's MLIR export, i.e. the "
             "`outfile` the clang plugin passes; with EXPORT_REACTANT set the "
             "raised module is written to <path>.mlir"),
    cl::value_desc("path"), cl::init(""));

extern "C" void registerReactant(PassBuilder &PB,
                                 std::vector<std::string> gpubinaries,
                                 std::string outfile);

extern "C" int optMain(int argc, char **argv,
                       ArrayRef<std::function<void(PassBuilder &)>>
                           PassBuilderCallbacks);

int main(int argc, char **argv) {
  // optMain parses the command line before running the callbacks, so the
  // options above are populated by the time this reads them.
  std::function<void(PassBuilder &)> plugins[] = {[](PassBuilder &PB) {
    registerReactant(
        PB, std::vector<std::string>(GPUBinaries.begin(), GPUBinaries.end()),
        ReactantOutfile);
  }};
  return optMain(argc, argv, plugins);
}
