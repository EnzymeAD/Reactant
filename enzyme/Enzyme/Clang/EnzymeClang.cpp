//===- EnzymeClang.cpp - Automatic Differentiation Transformation Pass ----===//
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
// This file contains a clang plugin for Enzyme.
//
//===----------------------------------------------------------------------===//

#include <limits>

#include "clang/AST/Attr.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

#include "Enzyme/Utils.h"

#include "bundled_includes.h"

using namespace clang;

#if LLVM_VERSION_MAJOR >= 18
constexpr auto StructKind = clang::TagTypeKind::Struct;
#else
constexpr auto StructKind = clang::TagTypeKind::TTK_Struct;
#endif

extern llvm::cl::opt<std::string> ReactantBackend;

std::vector<std::string> GlobalOptimizationRules;

struct TesseraArgTypeGlobalInfo {
  unsigned idx;
  QualType type;
  SourceLocation loc;
};

static std::vector<TesseraArgTypeGlobalInfo> TesseraArgTypeGlobals;

template <typename ConsumerType>
class EnzymeAction final : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new ConsumerType(CI));
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    llvm::errs() << " parse args action\n";
    llvm::errs() << " pa: " << CI.getFrontendOpts().ProgramAction << "\n";
    llvm::errs() << " args:\n";
    for (auto a : args)
      llvm::errs() << "+ arg: " << a << "\n";
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

static void emitOptimizationRules(Sema &S, std::vector<std::string> &Rules) {
  auto &AST = S.getASTContext();
  SourceLocation loc;
  DeclContext *declCtx = AST.getTranslationUnitDecl();

  // create global variable for each optimization string
  for (size_t i = 0, e = Rules.size(); i != e; ++i) {
    auto &Id = AST.Idents.get("__tessera_optimize_rule_" + std::to_string(i));
    auto VD = VarDecl::Create(AST, declCtx, loc, loc, &Id, AST.IntTy, nullptr,
                              SC_Static);
    VD->setImplicit(true);
    VD->setInit(
        IntegerLiteral::Create(AST, llvm::APInt(32, 0), AST.IntTy, loc));
    VD->addAttr(clang::UsedAttr::CreateImplicit(AST));
    VD->addAttr(AnnotateAttr::CreateImplicit(
        AST, ("tessera_optimize=" + Rules[i]).c_str(), nullptr, 0));
    declCtx->addDecl(VD);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
  }
}

static void
emitTesseraArgTypeGlobals(Sema &S,
                          std::vector<TesseraArgTypeGlobalInfo> &Globals) {
  auto &AST = S.getASTContext();
  DeclContext *declCtx = AST.getTranslationUnitDecl();
  for (auto &info : Globals) {
    auto &Id = AST.Idents.get("__tessera_arg_type_" + std::to_string(info.idx));
    auto *VD = VarDecl::Create(AST, declCtx, info.loc, info.loc, &Id, info.type,
                               nullptr, SC_Static);
    VD->setImplicit(true);
    VD->setInit(new (AST) ImplicitValueInitExpr(info.type));
    VD->addAttr(clang::UsedAttr::CreateImplicit(AST));
    declCtx->addDecl(VD);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
  }
}

void MakeGlobalOfFn(FunctionDecl *FD, CompilerInstance &CI) {
  // if (FD->isLateTemplateParsed()) return;
  // TODO save any type info into string like attribute
}

struct Visitor : public RecursiveASTVisitor<Visitor> {
  CompilerInstance &CI;
  Visitor(CompilerInstance &CI) : CI(CI) {}
  bool VisitFunctionDecl(FunctionDecl *FD) {
    MakeGlobalOfFn(FD, CI);
    return true;
  }
};

extern "C" void registerReactant(llvm::PassBuilder &PB,
                                 std::vector<std::string> gpubins,
                                 std::string outfile);

extern "C" void registerExporter(llvm::PassBuilder &PB, std::string file);

class EnzymePlugin final : public clang::ASTConsumer {
  clang::CompilerInstance &CI;

public:
  EnzymePlugin(clang::CompilerInstance &CI) : CI(CI) {
    // Allow the wrapper to act as a plain clang: skip registering the
    // Reactant pass pipeline entirely.
    if (getenv("NO_REACTANT_PLUGIN"))
      return;
    // FrontendOptions &Opts = CI.getFrontendOpts();
    CodeGenOptions &CGOpts = CI.getCodeGenOpts();
    auto PluginName = "ClangReactant-" + std::to_string(LLVM_VERSION_MAJOR);
    // bool contains = false;

    if (StringRef(ReactantBackend.getValue()).starts_with("xla")) {
      llvm::errs() << " note: you need to add -lReactantExtra\n";
    }
    std::string inFile;
    for (auto in : CI.getFrontendOpts().Inputs) {
      if (in.isFile()) {
        inFile = in.getFile().str();
        llvm::errs() << " in: " << in.getFile() << "\n";
      }
    }
    if (CI.getLangOpts().CUDAIsDevice) {
      std::string file = CI.getFrontendOpts().OutputFile;
      file = inFile;
      CGOpts.PassBuilderCallbacks.push_back(
          [=](llvm::PassBuilder &PB) { registerExporter(PB, file); });
    } else {
      std::vector<std::string> gpubins;
      if (CGOpts.CudaGpuBinaryFileName.size()) {
        if (inFile.size())
          gpubins.push_back(inFile);
        // gpubins.push_back(CGOpts.CudaGpuBinaryFileName);
      }
      std::string file = CI.getFrontendOpts().OutputFile;
      CGOpts.PassBuilderCallbacks.push_back(
          [=](llvm::PassBuilder &PB) { registerReactant(PB, gpubins, file); });
    }

    CI.getPreprocessorOpts().Includes.push_back("/enzyme/enzyme/version");

    std::string PredefineBuffer;
    PredefineBuffer.reserve(4080);
    llvm::raw_string_ostream Predefines(PredefineBuffer);
    Predefines << CI.getPreprocessor().getPredefines();
    MacroBuilder Builder(Predefines);
    Builder.defineMacro("ENZYME_VERSION_MAJOR",
                        std::to_string(ENZYME_VERSION_MAJOR));
    Builder.defineMacro("ENZYME_VERSION_MINOR",
                        std::to_string(ENZYME_VERSION_MINOR));
    Builder.defineMacro("ENZYME_VERSION_PATCH",
                        std::to_string(ENZYME_VERSION_PATCH));
    Builder.defineMacro("REACTANT_BACKEND",
                        "\"" + ReactantBackend.getValue() + "\"");
    StringRef rbackend = ReactantBackend.getValue();
    if (rbackend.starts_with("xla")) {
      StringRef device = rbackend.drop_front(3);
      device.consume_front("-");
      Builder.defineMacro("REACTANT_XLA_BACKEND", "\"" + device.str() + "\"");
    }
    CI.getPreprocessor().setPredefines(Predefines.str());

    auto baseFS = &CI.getFileManager().getVirtualFileSystem();
    llvm::vfs::OverlayFileSystem *fuseFS(
        new llvm::vfs::OverlayFileSystem(baseFS));
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs(
        new llvm::vfs::InMemoryFileSystem());

    struct tm y2k = {};

    y2k.tm_hour = 0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;
    time_t timer = mktime(&y2k);
    for (const auto &pair : include_headers) {
      fs->addFile(StringRef(pair[0]), timer,
                  llvm::MemoryBuffer::getMemBuffer(
                      StringRef(pair[1]), StringRef(pair[0]),
                      /*RequiresNullTerminator*/ true));
    }

    fuseFS->pushOverlay(fs);
    fuseFS->pushOverlay(baseFS);
    CI.getFileManager().setVirtualFileSystem(fuseFS);

    auto DE = CI.getFileManager().getDirectoryRef("/enzymeroot");
    assert(DE);
    auto DL = DirectoryLookup(*DE, SrcMgr::C_User,
                              /*isFramework=*/false);
    CI.getPreprocessor().getHeaderSearchInfo().AddSearchPath(DL,
                                                             /*isAngled=*/true);
  }
  ~EnzymePlugin() {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Sema &S = CI.getSema();
    emitOptimizationRules(S, GlobalOptimizationRules);
    emitTesseraArgTypeGlobals(S, TesseraArgTypeGlobals);
  }
};

// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<EnzymeAction<EnzymePlugin>>
    X("enzyme", "Enzyme Plugin");

#if LLVM_VERSION_MAJOR > 10
namespace {

static bool ExpectForStatement(Sema &S, const ParsedAttr &Attr,
                               const Stmt *St) {
  if (!isa<ForStmt>(St)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr << Attr.isRegularKeywordAttribute() << ExpectedForLoopStatement;
    return false;
  }
  return true;
}

static void emitFunctionCall(Sema &S, Stmt *St, std::string FunctionName,
                             llvm::ArrayRef<uint64_t> argValues) {
  auto &AST = S.getASTContext();
  SourceLocation loc;

  DeclContext *declCtx = S.getCurLexicalContext();
  for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
    if (tmpCtx->isRecord()) {
      declCtx = tmpCtx->getParent();
    }
  }

  // create global variable at translation unit level
  auto &Id = AST.Idents.get(FunctionName);

  std::vector<QualType> ParamTypes(argValues.size(), AST.getNSUIntegerType());
  auto FunctionType = AST.getFunctionType(AST.VoidTy, ParamTypes, {});

  DeclarationName name(&Id);
  DeclarationNameInfo nameInfo(name, loc);
  StorageClass SC = SC_PrivateExtern;
  FunctionDecl *F = FunctionDecl::Create(
      AST, declCtx, loc, nameInfo, FunctionType, nullptr, SC, false, false,
      false, ConstexprSpecKind::Unspecified, {});
  SmallVector<ParmVarDecl *> Params;
  for (size_t i = 0; i < argValues.size(); i++) {
    auto &ParamName =
        AST.Idents.get(i == 0 ? "enable" : ("arg" + std::to_string(i)));
    auto P =
        ParmVarDecl::Create(AST, F, loc, loc, &ParamName,
                            AST.getNSUIntegerType(), nullptr, SC_None, nullptr);
    Params.push_back(P);
  }
  F->setParams(Params);
  F->setStorageClass(SC);
  F->addAttr(clang::UsedAttr::CreateImplicit(AST));

  S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(F));

  TemplateArgumentListInfo *TemplateArgs = nullptr;

  auto rval = ExprValueKind::VK_PRValue;

  auto ForSt = cast<ForStmt>(St);
  Stmt *body = ForSt->getBody();

  SmallVector<Stmt *> Stmts;

  auto FT = AST.getPointerType(F->getType());
  auto DR = DeclRefExpr::Create(
      AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(F), false, loc,
      F->getType(), ExprValueKind::VK_LValue, cast<NamedDecl>(F), TemplateArgs);
  Expr *expr =
      ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay, DR,
                               nullptr, rval, FPOptionsOverride());

  SmallVector<Expr *> Args;
  for (uint64_t argValue : argValues) {
    Args.push_back(IntegerLiteral::Create(AST, llvm::APInt(64, argValue),
                                          AST.getNSUIntegerType(), loc));
  }
  auto BO = CallExpr::Create(AST, expr, Args, F->getType(), rval, loc, {});

  Stmts.push_back(BO);
  Stmts.push_back(body);

  CompoundStmt *newBody = CompoundStmt::Create(AST, Stmts, {}, loc, loc);
  ForSt->setBody(newBody);
}

struct EnzymeLoopMincutSetAttrInfo : public ParsedAttrInfo {
  EnzymeLoopMincutSetAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "enzyme_set_mincut"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "enzyme_set_mincut"},
#else
        {ParsedAttr::AS_C2x, "enzyme_set_mincut"},
#endif
        {ParsedAttr::AS_CXX11, "enzyme_set_mincut"},
        {ParsedAttr::AS_CXX11, "enzyme::set_mincut"}};
    Spellings = S;
  }

  bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr,
                            const Stmt *St) const override {
    return ExpectForStatement(S, Attr, St);
  }

  AttrHandling handleStmtAttribute(Sema &S, Stmt *St, const ParsedAttr &Attr,
                                   class Attr *&Result) const override {
    if (Attr.getNumArgs() < 1 || Attr.getNumArgs() > 1) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_set_mincut' takes a single argument");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto *Arg0 = Attr.getArgAsExpr(0);
    clang::Expr::EvalResult EvalRes;

    if (!Arg0->EvaluateAsInt(EvalRes, S.getASTContext())) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "argument to 'enzyme_set_mincut' must be an "
          "integer constant");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    uint64_t enable = EvalRes.Val.getInt().getZExtValue();

    emitFunctionCall(S, St, "__enzyme_set_mincut", {enable});
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeLoopMincutSetAttrInfo>
    X2("enzyme_set_mincut", "");

struct EnzymeLoopCheckpointingEnableAttrInfo : public ParsedAttrInfo {
  EnzymeLoopCheckpointingEnableAttrInfo() {
    OptArgs = 2;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "enzyme_checkpointing_enable"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "enzyme_checkpointing_enable"},
#else
        {ParsedAttr::AS_C2x, "enzyme_checkpointing_enable"},
#endif
        {ParsedAttr::AS_CXX11, "enzyme_checkpointing_enable"},
        {ParsedAttr::AS_CXX11, "enzyme::checkpointing_enable"}};
    Spellings = S;
  }

  bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr,
                            const Stmt *St) const override {
    return ExpectForStatement(S, Attr, St);
  }

  AttrHandling handleStmtAttribute(Sema &S, Stmt *St, const ParsedAttr &Attr,
                                   class Attr *&Result) const override {
    unsigned NumArgs = Attr.getNumArgs();
    if (NumArgs > 2) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_checkpointing_enable' takes at most two arguments "
          "(a mode string and an optional integer)");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    uint64_t Mode = 1; // default: regular
    if (NumArgs >= 1) {
      auto *Arg0 = Attr.getArgAsExpr(0);
      StringLiteral *Literal =
          dyn_cast<StringLiteral>(Arg0->IgnoreParenCasts());
      if (!Literal) {
        unsigned ID = S.getDiagnostics().getCustomDiagID(
            DiagnosticsEngine::Error,
            "first argument to 'enzyme_checkpointing_enable' must be a "
            "string literal, either \"binomial\" or \"regular\"");
        S.Diag(Attr.getLoc(), ID);
        return AttributeNotApplied;
      }
      StringRef Mode0 = Literal->getString();
      if (Mode0 == "binomial") {
        Mode = 2;
      } else if (Mode0 == "regular") {
        Mode = 1;
      } else {
        unsigned ID = S.getDiagnostics().getCustomDiagID(
            DiagnosticsEngine::Error,
            "unknown checkpointing mode '%0', expected \"binomial\" or "
            "\"regular\"");
        S.Diag(Attr.getLoc(), ID) << Mode0;
        return AttributeNotApplied;
      }
    }

    // __enzyme_set_checkpointing is declared afresh (bypassing normal
    // redeclaration merging) at every attributed loop, so every call site
    // must agree on the same arity -- otherwise codegen can reuse an
    // earlier, differently-typed declaration for a later call, producing
    // invalid IR. Always emit both arguments, defaulting the count to a
    // sentinel (all bits set) when the caller didn't provide one, since 0
    // is a plausible real count.
    uint64_t Count = std::numeric_limits<uint64_t>::max();
    if (NumArgs >= 2) {
      auto *Arg1 = Attr.getArgAsExpr(1);
      clang::Expr::EvalResult EvalRes;
      if (!Arg1->EvaluateAsInt(EvalRes, S.getASTContext())) {
        unsigned ID = S.getDiagnostics().getCustomDiagID(
            DiagnosticsEngine::Error,
            "second argument to 'enzyme_checkpointing_enable' must be an "
            "integer constant");
        S.Diag(Attr.getLoc(), ID);
        return AttributeNotApplied;
      }
      Count = EvalRes.Val.getInt().getZExtValue();
    }

    emitFunctionCall(S, St, "__enzyme_set_checkpointing", {Mode, Count});
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeLoopCheckpointingEnableAttrInfo>
    X3("enzyme_checkpointing_enable", "");

struct EnzymeFunctionLikeAttrInfo : public ParsedAttrInfo {
  EnzymeFunctionLikeAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "enzyme_function_like"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "enzyme_function_like"},
#else
        {ParsedAttr::AS_C2x, "enzyme_function_like"},
#endif
        {ParsedAttr::AS_CXX11, "enzyme_function_like"},
        {ParsedAttr::AS_CXX11, "enzyme::function_like"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 1) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_function' attribute requires a single string argument");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    auto *Arg0 = Attr.getArgAsExpr(0);
    StringLiteral *Literal = dyn_cast<StringLiteral>(Arg0->IgnoreParenCasts());
    if (!Literal) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "first argument to 'enzyme_function_like' "
                                    "attribute must be a string literal");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
#if LLVM_VERSION_MAJOR >= 12
    D->addAttr(AnnotateAttr::Create(
        S.Context, ("enzyme_function_like=" + Literal->getString()).str(),
        nullptr, 0, Attr.getRange()));
    return AttributeApplied;
#else
    auto FD = cast<FunctionDecl>(D);
    // if (FD->isLateTemplateParsed()) return;
    auto &AST = S.getASTContext();
    DeclContext *declCtx = FD->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = FD->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto Tinfo = nullptr;
    auto Tinfo0 = nullptr;
    auto FT = AST.getPointerType(FD->getType());
    auto CharTy = AST.getIntTypeForBitwidth(8, false);
    auto FD0 = FieldDecl::Create(AST, RD, loc, loc, /*Ud*/ nullptr, FT, Tinfo0,
                                 /*expr*/ nullptr, /*mutable*/ true,
                                 /*inclassinit*/ ICIS_NoInit);
    FD0->setAccess(AS_public);
    RD->addDecl(FD0);
    auto FD1 = FieldDecl::Create(
        AST, RD, loc, loc, /*Ud*/ nullptr, AST.getPointerType(CharTy), Tinfo0,
        /*expr*/ nullptr, /*mutable*/ true, /*inclassinit*/ ICIS_NoInit);
    FD1->setAccess(AS_public);
    RD->addDecl(FD1);
    RD->completeDefinition();
    assert(RD->getDefinition());
    auto &Id = AST.Idents.get("__enzyme_function_like_autoreg_" +
                              FD->getNameAsString());
    auto T = AST.getRecordType(RD);
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, T, Tinfo, SC_None);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), loc, FD, false,
                                  loc, FD->getType(), ExprValueKind::VK_LValue,
                                  FD, TemplateArgs);
    auto rval = ExprValueKind::VK_PRValue;
    StringRef cstr = Literal->getString();
    Expr *exprs[2] = {
        ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                 DR, nullptr, rval, FPOptionsOverride()),
        ImplicitCastExpr::Create(
            AST, AST.getPointerType(CharTy), CastKind::CK_ArrayToPointerDecay,
            StringLiteral::Create(
                AST, cstr, stringkind,
                /*Pascal*/ false,
                AST.getStringLiteralArrayType(CharTy, cstr.size()), loc),
            nullptr, rval, FPOptionsOverride())};
    auto IL = new (AST) InitListExpr(AST, loc, exprs, loc);
    V->setInit(IL);
    IL->setType(T);
    if (IL->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "use of attribute 'enzyme_function_like' "
                                    "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
#endif
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeFunctionLikeAttrInfo>
    X4("enzyme_function_like", "");

static ParsedAttrInfo::AttrHandling
handleTesseraOpAttribute(Sema &S, Decl *D, const ParsedAttr &Attr,
                         StringRef attrName) {
  if (Attr.getNumArgs() < 1) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Error,
        "'%0' attribute requires at least a string argument");
    S.Diag(Attr.getLoc(), ID) << attrName;
    return ParsedAttrInfo::AttributeNotApplied;
  }

  // Parse the first arg as the op string
  auto *Arg0 = Attr.getArgAsExpr(0);
  StringLiteral *Literal = dyn_cast<StringLiteral>(Arg0->IgnoreParenCasts());
  if (!Literal) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Error, "first argument to '%0' "
                                  "attribute must be a string literal");
    S.Diag(Attr.getLoc(), ID) << attrName;
    return ParsedAttrInfo::AttributeNotApplied;
  }

  // Scan for val=in, val=out, and val=inout argument positions
  StringRef opStr = Literal->getString();
  bool hasArgList = opStr.contains('(');
  StringRef argList = opStr.slice(opStr.find('(') + 1, opStr.find(')'));
  SmallVector<unsigned> positionsToLift;
  unsigned numListedArgs = 0;
  if (!argList.trim().empty()) {
    SmallVector<StringRef> argParts;
    argList.split(argParts, ',');
    numListedArgs = argParts.size();
    for (auto [idx, arg] : llvm::enumerate(argParts)) {
      arg = arg.trim();
      StringRef marker = arg.split(':').second.trim();
      if (marker.starts_with("val="))
        positionsToLift.push_back(idx);
    }
  }

  // Emit a global for each marked parameter
  auto FD = cast<FunctionDecl>(D);
  DeclContext *declCtx = D->getDeclContext();
  for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
    if (tmpCtx->isRecord()) {
      declCtx = tmpCtx->getParent();
    }
  }
  auto params = FD->parameters();
  auto loc = FD->getLocation();

  // A non-static C++ member function receives the object as an implicit
  // leading `this` pointer, which is argument 0 of the emitted LLVM function
  // but is absent from FD->parameters(). Positions in the op string name the
  // call's arguments, so `this` occupies position 0 and the explicit
  // parameters shift over by one:
  //
  //   struct Mat {
  //     [[tessera::op("mfem.mult(this:val=in, x, y:val=out)")]]
  //     void Mult(const Vec &x, Vec &y) const;
  //   };
  const auto *MD = dyn_cast<CXXMethodDecl>(FD);
  bool hasImplicitThis = MD && MD->isInstance();

  // The lowering requires one entry in the arg list per function argument
  // (`this` included).
  unsigned numExpectedArgs = params.size() + (hasImplicitThis ? 1 : 0);
  if (hasArgList && numListedArgs != numExpectedArgs) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "'%0' argument list names %1 argument(s) but %2 takes %3%4; positions "
        "in the argument list must match the function's arguments one for one");
    S.Diag(Attr.getLoc(), ID)
        << attrName << numListedArgs << FD << numExpectedArgs
        << (hasImplicitThis ? " (counting the implicit 'this')" : "");
  }

  static unsigned globalCounter = 0;
  SmallVector<unsigned> liftedArgGlobalIndices;

  for (unsigned idx : positionsToLift) {
    QualType pointeeTy;
    if (hasImplicitThis && idx == 0) {
      // The pointee of `this` is the (possibly const-qualified) class type.
      pointeeTy = MD->getThisType()->getPointeeType();
    } else {
      unsigned paramIdx = idx - (hasImplicitThis ? 1 : 0);
      if (paramIdx >= params.size())
        continue;
      pointeeTy = params[paramIdx]->getType();
      if (pointeeTy->isPointerType() || pointeeTy->isReferenceType())
        pointeeTy = (pointeeTy->getPointeeType());
    }

    pointeeTy = pointeeTy.getUnqualifiedType();

    unsigned thisIdx = globalCounter++;
    liftedArgGlobalIndices.push_back(thisIdx);
    TesseraArgTypeGlobals.push_back({thisIdx, pointeeTy, loc});
  }

  // Build annotation string: "tessera_op=eigen.inv(x:val=in, y):3,4"
  std::string annotation = (attrName + "=" + opStr).str();

  // Parse remaining args representing sizes of function parameters
  for (auto [i, idx] : llvm::enumerate(liftedArgGlobalIndices)) {
    annotation += (i == 0 ? ":globals=" : ",") + std::to_string(idx);
  }

  D->addAttr(
      AnnotateAttr::Create(S.Context, annotation, nullptr, 0, Attr.getRange()));
  return ParsedAttrInfo::AttributeApplied;
}

struct TesseraOpAttrInfo : public ParsedAttrInfo {
  TesseraOpAttrInfo() {
    OptArgs = 15;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {{ParsedAttr::AS_GNU, "tessera_op"},
#if LLVM_VERSION_MAJOR > 17
                                     {ParsedAttr::AS_C23, "tessera_op"},
#else
                                     {ParsedAttr::AS_C2x, "tessera_op"},
#endif
                                     {ParsedAttr::AS_CXX11, "tessera_op"},
                                     {ParsedAttr::AS_CXX11, "tessera::op"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    return handleTesseraOpAttribute(S, D, Attr, "tessera_op");
  }
};

static ParsedAttrInfoRegistry::Add<TesseraOpAttrInfo> T1("tessera_op", "");

struct PureTesseraOpAttrInfo : public ParsedAttrInfo {
  PureTesseraOpAttrInfo() {
    OptArgs = 15;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "pure_tessera_op"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "pure_tessera_op"},
#else
        {ParsedAttr::AS_C2x, "pure_tessera_op"},
#endif
        {ParsedAttr::AS_CXX11, "pure_tessera_op"},
        {ParsedAttr::AS_CXX11, "tessera::pure_op"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    return handleTesseraOpAttribute(S, D, Attr, "pure_tessera_op");
  }
};

static ParsedAttrInfoRegistry::Add<PureTesseraOpAttrInfo> T2("pure_tessera_op",
                                                             "");

struct EnzymeShouldRecomputeAttrInfo : public ParsedAttrInfo {
  EnzymeShouldRecomputeAttrInfo() {
    OptArgs = 1;
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "enzyme_shouldrecompute"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "enzyme_shouldrecompute"},
#else
        {ParsedAttr::AS_C2x, "enzyme_shouldrecompute"},
#endif
        {ParsedAttr::AS_CXX11, "enzyme_shouldrecompute"},
        {ParsedAttr::AS_CXX11, "enzyme::shouldrecompute"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    if (auto VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return true;
    }
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions and globals";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_inactive' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    D->addAttr(AnnotateAttr::Create(S.Context, "enzyme_shouldrecompute",
                                    nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeShouldRecomputeAttrInfo>
    ESR("enzyme_shouldrecompute", "");

struct EnzymeInactiveAttrInfo : public ParsedAttrInfo {
  EnzymeInactiveAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "enzyme_inactive"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "enzyme_inactive"},
#else
        {ParsedAttr::AS_C2x, "enzyme_inactive"},
#endif
        {ParsedAttr::AS_CXX11, "enzyme_inactive"},
        {ParsedAttr::AS_CXX11, "enzyme::inactive"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    if (auto VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return true;
    }
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions and globals";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_inactive' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto &AST = S.getASTContext();
    DeclContext *declCtx = D->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = D->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto T = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getType()
                                  : cast<VarDecl>(D)->getType();
    auto Name = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getNameAsString()
                                     : cast<VarDecl>(D)->getNameAsString();
    auto FT = AST.getPointerType(T);
    auto subname = isa<FunctionDecl>(D) ? "inactivefn" : "inactive_global";
    auto &Id = AST.Idents.get(
        (StringRef("__enzyme_") + subname + "_autoreg_" + Name).str());
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, FT, nullptr, SC_None);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(D), false, loc, T,
        ExprValueKind::VK_LValue, cast<NamedDecl>(D), TemplateArgs);
    auto rval = ExprValueKind::VK_PRValue;
    Expr *expr = nullptr;
    if (isa<FunctionDecl>(D)) {
      expr =
          ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                   DR, nullptr, rval, FPOptionsOverride());
    } else {
      expr =
          UnaryOperator::Create(AST, DR, UnaryOperatorKind::UO_AddrOf, FT, rval,
                                clang::ExprObjectKind ::OK_Ordinary, loc,
                                /*canoverflow*/ false, FPOptionsOverride());
    }

    if (expr->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "use of attribute 'enzyme_inactive' "
                                    "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    V->setInit(expr);
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeInactiveAttrInfo> X5("enzyme_inactive",
                                                              "");

struct EnzymeNoFreeAttrInfo : public ParsedAttrInfo {
  EnzymeNoFreeAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {{ParsedAttr::AS_GNU, "enzyme_nofree"},
#if LLVM_VERSION_MAJOR > 17
                                     {ParsedAttr::AS_C23, "enzyme_nofree"},
#else
                                     {ParsedAttr::AS_C2x, "enzyme_nofree"},
#endif
                                     {ParsedAttr::AS_CXX11, "enzyme_nofree"},
                                     {ParsedAttr::AS_CXX11, "enzyme::nofree"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    if (auto VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return true;
    }
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions and globals";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_nofree' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto &AST = S.getASTContext();
    DeclContext *declCtx = D->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = D->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto T = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getType()
                                  : cast<VarDecl>(D)->getType();
    auto Name = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getNameAsString()
                                     : cast<VarDecl>(D)->getNameAsString();
    auto FT = AST.getPointerType(T);
    auto &Id = AST.Idents.get(
        (StringRef("__enzyme_nofree") + "_autoreg_" + Name).str());
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, FT, nullptr, SC_None);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(D), false, loc, T,
        ExprValueKind::VK_LValue, cast<NamedDecl>(D), TemplateArgs);
    auto rval = ExprValueKind::VK_PRValue;
    Expr *expr = nullptr;
    if (isa<FunctionDecl>(D)) {
      expr =
          ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                   DR, nullptr, rval, FPOptionsOverride());
    } else {
      expr =
          UnaryOperator::Create(AST, DR, UnaryOperatorKind::UO_AddrOf, FT, rval,
                                clang::ExprObjectKind ::OK_Ordinary, loc,
                                /*canoverflow*/ false, FPOptionsOverride());
    }

    if (expr->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "use of attribute 'enzyme_nofree' "
                                    "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    V->setInit(expr);
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeNoFreeAttrInfo> X6("enzyme_nofree",
                                                            "");

struct EnzymeSparseAccumulateAttrInfo : public ParsedAttrInfo {
  EnzymeSparseAccumulateAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "enzyme_sparse_accumulate"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "enzyme_sparse_accumulate"},
#else
        {ParsedAttr::AS_C2x, "enzyme_sparse_accumulate"},
#endif
        {ParsedAttr::AS_CXX11, "enzyme_sparse_accumulate"},
        {ParsedAttr::AS_CXX11, "enzyme::sparse_accumulate"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_sparse_accumulate' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto &AST = S.getASTContext();
    DeclContext *declCtx = D->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = D->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto T = cast<FunctionDecl>(D)->getType();
    auto Name = cast<FunctionDecl>(D)->getNameAsString();
    auto FT = AST.getPointerType(T);
    auto &Id = AST.Idents.get(
        (StringRef("__enzyme_sparse_accumulate") + "_autoreg_" + Name).str());
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, FT, nullptr, SC_None);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(D), false, loc, T,
        ExprValueKind::VK_LValue, cast<NamedDecl>(D), TemplateArgs);
    auto rval = ExprValueKind::VK_PRValue;
    Expr *expr = nullptr;
    expr =
        ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                 DR, nullptr, rval, FPOptionsOverride());

    if (expr->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "use of attribute 'enzyme_sparse_accumulate' "
          "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    V->setInit(expr);
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeSparseAccumulateAttrInfo>
    SparseX("enzyme_sparse_accumulate", "");

// #pragma optimize "expression1 -> expression2"
class PragmaTesseraOptimizeHandler : public PragmaHandler {
public:
  PragmaTesseraOptimizeHandler() : PragmaHandler("optimize") {}
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &Tok) override {
    PP.Lex(Tok);
    if (Tok.isNot(tok::string_literal)) {
      PP.Diag(Tok.getLocation(), diag::err_expected) << tok::string_literal;
      return;
    }

    std::string OptimizationRule;
    if (!PP.FinishLexStringLiteral(Tok, OptimizationRule, "pragma optimize",
                                   /*AllowMacroExpansion=*/false))
      return;
    if (Tok.isNot(tok::eod)) {
      PP.Diag(Tok, diag::ext_pp_extra_tokens_at_eol)
          << "pragma optimize warning";
      return;
    }
    GlobalOptimizationRules.push_back(OptimizationRule);
  }
};

static PragmaHandlerRegistry::Add<PragmaTesseraOptimizeHandler>
    OptX("optimize", "custom tessera optimization");
} // namespace

#endif
