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

#include <functional>

using namespace llvm;

extern "C" void registerReactant2(llvm::PassBuilder &PB);

extern "C" int optMain(int argc, char **argv,
                       llvm::ArrayRef<std::function<void(llvm::PassBuilder &)>>
                           PassBuilderCallbacks);

int main(int argc, char **argv) {
  std::function<void(llvm::PassBuilder &)> plugins[] = {registerReactant2};
  return optMain(argc, argv, plugins);
}
