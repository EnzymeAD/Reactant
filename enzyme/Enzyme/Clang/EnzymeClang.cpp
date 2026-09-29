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
#include <type_traits>
#include <utility>

#include "clang/AST/Attr.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/CharInfo.h"
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

#include "Enzyme/Utils.h"

#include "bundled_includes.h"

using namespace clang;

#if LLVM_VERSION_MAJOR >= 18
constexpr auto StructKind = clang::TagTypeKind::Struct;
#else
constexpr auto StructKind = clang::TagTypeKind::TTK_Struct;
#endif

extern llvm::cl::opt<std::string> ReactantBackend;

static llvm::cl::opt<bool> DisableReactant(
    "disable-reactant", llvm::cl::init(false), llvm::cl::Hidden,
    llvm::cl::desc("Act as a plain clang: do not register the Reactant "
                   "pass pipeline; DISABLE_REACTANT in the environment "
                   "does the same when the flag is not given"));

// clang renamed CodeGenOptions::CudaGpuBinaryFileName to
// OffloadBinaryToEmbedFile (llvm/llvm-project#216090) within one major
// version, so the member's presence rather than LLVM_VERSION_MAJOR decides.
template <typename T, typename = void>
struct HasOffloadBinaryToEmbedFile : std::false_type {};
template <typename T>
struct HasOffloadBinaryToEmbedFile<
    T, std::void_t<decltype(std::declval<T &>().OffloadBinaryToEmbedFile)>>
    : std::true_type {};

template <typename T>
static const std::string &offloadBinaryToEmbedFile(const T &Opts) {
  if constexpr (HasOffloadBinaryToEmbedFile<T>::value)
    return Opts.OffloadBinaryToEmbedFile;
  else
    return Opts.CudaGpuBinaryFileName;
}

std::vector<std::string> GlobalOptimizationRules;

struct TesseraArgTypeGlobalInfo {
  unsigned idx;
  QualType type;
  SourceLocation loc;
};

static std::vector<TesseraArgTypeGlobalInfo> TesseraArgTypeGlobals;

// Tessera ops whose declaration must be forced into the module; see
// emitTesseraOpRefGlobals.
static std::vector<FunctionDecl *> TesseraOpFunctions;

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

/// The value of the C name `Name` in the syntax of a rule, if it has one that
/// a rule can hold: an enumerator becomes its value, and a macro that expands
/// to a single integer or string becomes that literal. A macro that expands to
/// another name is followed.
static std::optional<std::string> resolveRuleName(Sema &S, StringRef Name,
                                                  unsigned Depth = 0) {
  if (Depth > 8)
    return std::nullopt;
  Preprocessor &PP = S.getPreprocessor();
  IdentifierInfo *II = PP.getIdentifierInfo(Name);

  if (const MacroInfo *MI = PP.getMacroInfo(II)) {
    if (!MI->isObjectLike() || MI->getNumTokens() != 1)
      return std::nullopt;
    const Token &T = MI->getReplacementToken(0);
    std::string Spelling = PP.getSpelling(T);
    switch (T.getKind()) {
    case tok::string_literal: {
      // Only a plain "..." with nothing in it a rule string cannot hold.
      StringRef Text(Spelling);
      if (!Text.consume_front("\"") || !Text.consume_back("\"") ||
          Text.find_first_of("\\'") != StringRef::npos)
        return std::nullopt;
      return "'" + Text.str() + "'";
    }
    case tok::numeric_constant: {
      int64_t Value;
      if (StringRef(Spelling).getAsInteger(0, Value))
        return std::nullopt;
      return std::to_string(Value);
    }
    case tok::identifier:
      return resolveRuleName(S, Spelling, Depth + 1);
    default:
      return std::nullopt;
    }
  }

  for (NamedDecl *D :
       S.getASTContext().getTranslationUnitDecl()->lookup(DeclarationName(II)))
    if (auto *ECD = dyn_cast<EnumConstantDecl>(D))
      return toString(ECD->getInitVal(), 10);
  return std::nullopt;
}

/// Replace the C names on the right-hand side of `Rule` with their values, so
/// that a rule written in a library's header can use the library's own names
/// for its constants, as `MAT_SPD` or `KSPCG`: Tessera sees the rule long after
/// those names are gone, and only understands numbers and strings.
///
/// Only a name the left-hand side does not bind is looked up, which is one the
/// rule could not otherwise use, so no rule that worked before changes. A name
/// that is part of a call, as the `petsc` and `ksp_set_type` of
/// `petsc.ksp_set_type(...)`, is left alone, and so is one with no value: the
/// rule parser reports it as unbound.
static std::string resolveRuleNames(Sema &S, StringRef Rule) {
  size_t Arrow = Rule.find("->");
  if (Arrow == StringRef::npos)
    return Rule.str();

  // Numbers are skipped whole, so that the exponent of `1e3` is not taken for
  // a name.
  auto skipNumber = [&](size_t I) {
    size_t J = I;
    while (J < Rule.size() &&
           (isAsciiIdentifierContinue(Rule[J]) || Rule[J] == '.' ||
            ((Rule[J] == '-' || Rule[J] == '+') &&
             (Rule[J - 1] == 'e' || Rule[J - 1] == 'E'))))
      ++J;
    return J;
  };
  auto skipName = [&](size_t I) {
    size_t J = I;
    while (J < Rule.size() && isAsciiIdentifierContinue(Rule[J]))
      ++J;
    return J;
  };

  // Every name before the arrow. Besides the variables the rule binds, this
  // takes in op and predicate names, which is harmless: they are never
  // looked up anyway.
  llvm::StringSet<> Bound;
  for (size_t I = 0; I < Arrow;) {
    if (isDigit(Rule[I]))
      I = skipNumber(I);
    else if (isAsciiIdentifierStart(Rule[I])) {
      size_t J = skipName(I);
      Bound.insert(Rule.slice(I, J));
      I = J;
    } else
      ++I;
  }

  std::string Out = Rule.take_front(Arrow + 2).str();
  for (size_t I = Arrow + 2, N = Rule.size(); I < N;) {
    char C = Rule[I];
    if (C == '\'') {
      size_t J = Rule.find('\'', I + 1);
      J = J == StringRef::npos ? N : J + 1;
      Out += Rule.slice(I, J);
      I = J;
      continue;
    }
    if (isDigit(C)) {
      size_t J = skipNumber(I);
      Out += Rule.slice(I, J);
      I = J;
      continue;
    }
    if (!isAsciiIdentifierStart(C)) {
      Out += C;
      ++I;
      continue;
    }

    size_t J = skipName(I);
    StringRef Name = Rule.slice(I, J);
    size_t Before = I, After = J;
    while (Before > 0 && isWhitespace(Rule[Before - 1]))
      --Before;
    while (After < N && isWhitespace(Rule[After]))
      ++After;
    bool PartOfCall = (Before > 0 && Rule[Before - 1] == '.') ||
                      (After < N && (Rule[After] == '.' || Rule[After] == '('));

    std::optional<std::string> Value;
    if (!PartOfCall && !Bound.contains(Name))
      Value = resolveRuleName(S, Name);
    Out += Value ? *Value : Name.str();
    I = J;
  }
  return Out;
}

static void emitOptimizationRules(Sema &S, std::vector<std::string> &Rules) {
  auto &AST = S.getASTContext();
  SourceLocation loc;
  DeclContext *declCtx = AST.getTranslationUnitDecl();

  // Resolved here rather than when the pragma is read, since a rule may name
  // constants declared after it.
  for (std::string &Rule : Rules)
    Rule = resolveRuleNames(S, Rule);

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

// Emit a used global holding the function's address so that the declaration is
// created and clang emits the tessera annotation..
static void emitTesseraOpRefGlobals(Sema &S, std::vector<FunctionDecl *> &Fns) {
  auto &AST = S.getASTContext();
  DeclContext *declCtx = AST.getTranslationUnitDecl();
  unsigned counter = 0;

  for (auto *FD : Fns) {
    // A definition in this TU is already pinned by the `used` attribute.
    if (FD->hasBody() || FD->isDeleted())
      continue;
    // An uninstantiated template has no address to take.
    if (FD->getDescribedFunctionTemplate() || FD->isDependentContext())
      continue;

    const auto *MD = dyn_cast<CXXMethodDecl>(FD);
    // Constructors and destructors have no address, and a pointer to a virtual
    // member is a vtable index rather than a reference to the function, so
    // neither forces the declaration into the module.
    if (MD && (isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD) ||
               MD->isVirtual())) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Warning,
          "tessera op %0 has no definition here and cannot be emitted "
          "automatically; a call to it is needed for a tessera rewrite to "
          "reference it");
      S.Diag(FD->getLocation(), ID) << FD;
      continue;
    }

    auto loc = FD->getLocation();
    QualType refTy;
    if (MD && MD->isInstance()) {
#if LLVM_VERSION_MAJOR >= 23
      refTy = AST.getMemberPointerType(
          FD->getType(), /*Qualifier=*/std::nullopt, MD->getParent());
#elif LLVM_VERSION_MAJOR >= 21
      refTy = AST.getMemberPointerType(FD->getType(), /*Qualifier=*/nullptr,
                                       MD->getParent());
#else
      refTy = AST.getMemberPointerType(FD->getType(),
                                       MD->getParent()->getTypeForDecl());
#endif
    } else {
      refTy = AST.getPointerType(FD->getType());
    }

    // static <ptr type> __tessera_op_ref_N __attribute__((used)) = &f;
    auto *DR = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(FD), false, loc,
        FD->getType(), ExprValueKind::VK_LValue, cast<NamedDecl>(FD), nullptr);
    auto *Ref = UnaryOperator::Create(
        AST, DR, UnaryOperatorKind::UO_AddrOf, refTy, ExprValueKind::VK_PRValue,
        ExprObjectKind::OK_Ordinary, loc,
        /*CanOverflow=*/false, FPOptionsOverride());

    auto &Id = AST.Idents.get("__tessera_op_ref_" + std::to_string(counter++));
    auto *VD =
        VarDecl::Create(AST, declCtx, loc, loc, &Id, refTy, nullptr, SC_Static);
    VD->setImplicit(true);
    VD->setInit(Ref);
    VD->addAttr(clang::UsedAttr::CreateImplicit(AST));
    S.MarkFunctionReferenced(loc, FD);
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
    bool disable = DisableReactant;
    if (DisableReactant.getNumOccurrences() == 0)
      if (const char *env = getenv("DISABLE_REACTANT"))
        disable = env[0] && env[0] != '0';
    if (disable)
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
      if (offloadBinaryToEmbedFile(CGOpts).size()) {
        if (inFile.size())
          gpubins.push_back(inFile);
        // gpubins.push_back(offloadBinaryToEmbedFile(CGOpts));
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
    emitTesseraOpRefGlobals(S, TesseraOpFunctions);
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
  // A malformed tessera annotation only costs the rewrites it would have
  // allowed, so it is reported as a warning and left out, and the function
  // compiles as though it were not there.
  if (Attr.getNumArgs() < 1) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "'%0' attribute requires at least a string argument; it is ignored");
    S.Diag(Attr.getLoc(), ID) << attrName;
    return ParsedAttrInfo::AttributeNotApplied;
  }

  // Parse the first arg as the op string
  auto *Arg0 = Attr.getArgAsExpr(0);
  StringLiteral *Literal = dyn_cast<StringLiteral>(Arg0->IgnoreParenCasts());
  if (!Literal) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning, "first argument to '%0' attribute must be "
                                    "a string literal; it is ignored");
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
  // Lifting a function whose argument list does not line up with it would
  // produce a tessera.define that does not verify.
  if (hasArgList && numListedArgs != numExpectedArgs) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "'%0' argument list names %1 argument(s) but %2 takes %3%4; positions "
        "in the argument list must match the function's arguments one for "
        "one; it is ignored");
    S.Diag(Attr.getLoc(), ID)
        << attrName << numListedArgs << FD << numExpectedArgs
        << (hasImplicitThis ? " (counting the implicit 'this')" : "");
    return ParsedAttrInfo::AttributeNotApplied;
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

  auto &AST = S.getASTContext();
  D->addAttr(
      AnnotateAttr::Create(AST, annotation, nullptr, 0, Attr.getRange()));
  D->addAttr(clang::UsedAttr::CreateImplicit(AST));
  TesseraOpFunctions.push_back(FD);
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

// A property of a function's inputs or outputs: a name like "SPD" or
// "symmetric" that a rule's condition can test, `if SPD(A), ...`.
//
//   [[tessera::guarantees("SPD")]]         the return value is always SPD
//   [[tessera::guarantees("SPD(M)")]]      parameter M is always SPD once it
//                                          returns
//   [[tessera::assumes("SPD(A)", "symmetric(B)")]]
//       A is always SPD and B always symmetric when the function is called
//   [[tessera::preserves("SPD", "A", "B")]]
//       the output is SPD whenever A and B both are: the return value, or the
//       one parameter the tessera_op writes if it returns nothing
//   [[tessera::readonly("A")]]
//       the function does not change the object pointer A refers to
//
// A guarantee on a pointer the function takes as it is -- a handle, such as a
// PETSc Mat -- is about the object it refers to, from the call on:
//
//   PetscErrorCode FillCOO(Mat A, void *ctx)
//       __attribute__((tessera_guarantees("SPD(A)")));
//   PetscErrorCode MatMult(Mat A, Vec x, Vec y)
//       __attribute__((tessera_readonly("A")));
//
// Then a later call given A can rely on it being SPD, as long as every call
// given A in between is readonly in it.
//
// A guarantee or assumption may state several facts at once, one per string,
// and a readonly may name several parameters. Parameters are named as
// written, or by position; `this` names the object of a member function.
// Positions count as tessera_op argument lists do, `this` first, so each fact
// becomes one annotation such as "tessera_guarantees=SPD:arg2", a preserves
// becomes "tessera_preserves=SPD:0,1", and a readonly "tessera_readonly=arg0",
// which lift-tessera-annotations can line up with argModes.
//
// Only the GNU spelling __attribute__((tessera_guarantees(...))) and the
// unscoped [[tessera_guarantees(...)]] keep their arguments; the scoped
// [[tessera::guarantees(...)]] reaches the plugin with none, and is ignored
// with a warning.
enum class TesseraPropertyKind { Guarantees, Assumes, Preserves, Readonly };

// A property name: letters, digits and underscores.
static bool isTesseraPropertyName(StringRef name) {
  return !name.empty() && llvm::all_of(name, [](char c) {
    return llvm::isAlnum(c) || c == '_';
  });
}

// Split a fact written "SPD(M)" into the property and the parameter. A bare
// "SPD" has no parameter. Returns false if the text is neither form.
static bool parseTesseraFact(StringRef text, StringRef &property,
                             StringRef &param) {
  text = text.trim();
  size_t open = text.find('(');
  if (open == StringRef::npos) {
    property = text;
    param = "";
    return isTesseraPropertyName(property);
  }
  if (text.back() != ')')
    return false;
  property = text.take_front(open).trim();
  param = text.drop_front(open + 1).drop_back().trim();
  return isTesseraPropertyName(property) && !param.empty() &&
         !param.contains('(') && !param.contains(')') && !param.contains(',');
}

// The tessera_op argument position a parameter name refers to.
static std::optional<unsigned> getTesseraArgPosition(FunctionDecl *FD,
                                                     StringRef name) {
  const auto *MD = dyn_cast<CXXMethodDecl>(FD);
  bool hasImplicitThis = MD && MD->isInstance();
  unsigned numArgs = FD->getNumParams() + (hasImplicitThis ? 1 : 0);
  if (name == "this")
    return hasImplicitThis ? std::optional<unsigned>(0) : std::nullopt;
  unsigned position;
  if (!name.getAsInteger(10, position))
    return position < numArgs ? std::optional<unsigned>(position)
                              : std::nullopt;
  for (auto [i, param] : llvm::enumerate(FD->parameters()))
    if (param->getName() == name)
      return i + (hasImplicitThis ? 1 : 0);
  return std::nullopt;
}

static ParsedAttrInfo::AttrHandling
handleTesseraPropertyAttribute(Sema &S, Decl *D, const ParsedAttr &Attr,
                               TesseraPropertyKind kind) {
  bool isGuarantee = kind == TesseraPropertyKind::Guarantees;
  StringRef attrName =
      isGuarantee                              ? "tessera::guarantees"
      : kind == TesseraPropertyKind::Assumes   ? "tessera::assumes"
      : kind == TesseraPropertyKind::Preserves ? "tessera::preserves"
                                               : "tessera::readonly";
  // A malformed annotation is a mistake in what is known, not in the program,
  // so it only costs the facts it would have given: warn, and leave it out.
  auto warn = [&](const Twine &message) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Warning,
                                                     "%0; it is ignored");
    S.Diag(Attr.getLoc(), ID) << message.str();
  };

  if (Attr.getNumArgs() < (kind == TesseraPropertyKind::Preserves ? 2u : 1u)) {
    warn(isGuarantee ? "'tessera::guarantees' takes one or more facts such as "
                       "\"SPD\" or \"SPD(M)\""
         : kind == TesseraPropertyKind::Assumes
             ? "'tessera::assumes' takes one or more facts such as \"SPD(M)\""
         : kind == TesseraPropertyKind::Preserves
             ? "'tessera::preserves' takes a property name and the "
               "parameters it is preserved from"
             : "'tessera::readonly' takes the pointer parameters the function "
               "does not change the objects of");
    return ParsedAttrInfo::AttributeNotApplied;
  }

  SmallVector<std::string> args;
  for (unsigned i = 0, e = Attr.getNumArgs(); i != e; ++i) {
    auto *Literal =
        dyn_cast<StringLiteral>(Attr.getArgAsExpr(i)->IgnoreParenCasts());
    if (!Literal) {
      warn("arguments to '" + attrName + "' must be string literals");
      return ParsedAttrInfo::AttributeNotApplied;
    }
    args.push_back(Literal->getString().str());
  }

  auto *FD = cast<FunctionDecl>(D);
  SmallVector<std::string> annotations;
  if (kind == TesseraPropertyKind::Preserves) {
    // A preserves is used whole or not at all: leaving out an input it could
    // not name would claim the property survives with fewer inputs having it.
    StringRef property = args[0];
    if (!isTesseraPropertyName(property)) {
      warn("'" + property + "' is not a valid property name for '" + attrName +
           "'; use letters, digits and underscores");
      return ParsedAttrInfo::AttributeNotApplied;
    }
    std::string annotation = ("tessera_preserves=" + property + ":").str();
    for (unsigned i = 1; i < args.size(); ++i) {
      auto pos = getTesseraArgPosition(FD, args[i]);
      if (!pos) {
        warn("'tessera::preserves' names '" + args[i] +
             "', which is not a parameter of '" + FD->getNameAsString() + "'");
        return ParsedAttrInfo::AttributeNotApplied;
      }
      annotation += (i == 1 ? "" : ",") + std::to_string(*pos);
    }
    annotations.push_back(std::move(annotation));
  }

  // Each parameter a readonly names stands on its own too: one it cannot
  // name is left out, which only means calls are taken to change that object.
  if (kind == TesseraPropertyKind::Readonly) {
    const auto *MD = dyn_cast<CXXMethodDecl>(FD);
    unsigned thisOffset = MD && MD->isInstance() ? 1 : 0;
    for (StringRef param : args) {
      param = param.trim();
      auto pos = getTesseraArgPosition(FD, param);
      if (!pos) {
        warn("'tessera::readonly' names '" + param +
             "', which is not a parameter of '" + FD->getNameAsString() + "'");
        continue;
      }
      if (param != "this" && *pos >= thisOffset) {
        QualType type = FD->getParamDecl(*pos - thisOffset)->getType();
        if (!type->isPointerType() && !type->isReferenceType()) {
          warn("'tessera::readonly' names '" + param +
               "', which is not a pointer or reference, so there is no "
               "object to leave alone");
          continue;
        }
      }
      annotations.push_back(("tessera_readonly=arg" + Twine(*pos)).str());
    }
  }

  // Each argument of a guarantee or assumption is one fact, and stands on its
  // own, so a malformed one is left out and the rest are kept.
  for (StringRef text : args) {
    if (kind == TesseraPropertyKind::Preserves ||
        kind == TesseraPropertyKind::Readonly)
      break;
    StringRef property, param;
    if (!parseTesseraFact(text, property, param)) {
      warn("'" + text + "' is not a fact '" + attrName +
           "' understands; write a property name and the parameter it applies "
           "to, such as \"SPD(M)\"" +
           (isGuarantee ? ", or a bare \"SPD\" for the return value" : ""));
      continue;
    }

    if (isGuarantee && param == "return")
      param = "";
    if (param.empty()) {
      if (!isGuarantee) {
        warn("'tessera::assumes' needs the parameter each fact applies to; "
             "write \"" +
             property + "(M)\"");
        continue;
      }
      // The old form gave the parameter as a separate string, which would
      // now read as a second property of the return value.
      if (getTesseraArgPosition(FD, property)) {
        warn("'" + property + "' names a parameter of '" +
             FD->getNameAsString() +
             "', not a property; state a property of it as \"SPD(" + property +
             ")\"");
        continue;
      }
      if (FD->getReturnType()->isVoidType()) {
        warn("'tessera::guarantees' on a function returning void must name "
             "the parameter each fact applies to; write \"" +
             property + "(M)\"");
        continue;
      }
      annotations.push_back(
          ("tessera_guarantees=" + property + ":return").str());
      continue;
    }

    auto pos = getTesseraArgPosition(FD, param);
    if (!pos) {
      warn("'" + attrName + "' names '" + param +
           "', which is not a parameter of '" + FD->getNameAsString() + "'");
      continue;
    }
    if (isGuarantee) {
      // Only something the function writes through can come out of it with
      // a property it did not have going in.
      const auto *MD = dyn_cast<CXXMethodDecl>(FD);
      unsigned paramIdx = *pos - (MD && MD->isInstance() ? 1 : 0);
      if (param != "this" && paramIdx < FD->getNumParams()) {
        QualType type = FD->getParamDecl(paramIdx)->getType();
        if (!type->isPointerType() && !type->isReferenceType()) {
          warn("'tessera::guarantees' names '" + param +
               "', which is passed by value, so the function cannot give it a "
               "property");
          continue;
        }
      }
    }
    annotations.push_back(
        ((isGuarantee ? "tessera_guarantees=" : "tessera_assumes=") + property +
         ":arg" + Twine(*pos))
            .str());
  }
  if (annotations.empty())
    return ParsedAttrInfo::AttributeNotApplied;

  auto &AST = S.getASTContext();
  for (const std::string &annotation : annotations)
    D->addAttr(
        AnnotateAttr::Create(AST, annotation, nullptr, 0, Attr.getRange()));
  // A property is read off the call that produced a value, or off the
  // parameters of the function that assumes it, and a handle's off the calls
  // given it, so that call, or that function's own body, has to survive until
  // tessera-apply-pdl runs; the pipeline inlines long before then.
  if (!D->hasAttr<AlwaysInlineAttr>() && !D->hasAttr<NoInlineAttr>())
    D->addAttr(NoInlineAttr::CreateImplicit(AST));
  return ParsedAttrInfo::AttributeApplied;
}

template <TesseraPropertyKind Kind>
struct TesseraPropertyAttrInfo : public ParsedAttrInfo {
  TesseraPropertyAttrInfo() {
    OptArgs = 15;
    static constexpr Spelling GuaranteesSpellings[] = {
        {ParsedAttr::AS_GNU, "tessera_guarantees"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "tessera_guarantees"},
#else
        {ParsedAttr::AS_C2x, "tessera_guarantees"},
#endif
        {ParsedAttr::AS_CXX11, "tessera_guarantees"},
        {ParsedAttr::AS_CXX11, "tessera::guarantees"}};
    static constexpr Spelling AssumesSpellings[] = {
        {ParsedAttr::AS_GNU, "tessera_assumes"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "tessera_assumes"},
#else
        {ParsedAttr::AS_C2x, "tessera_assumes"},
#endif
        {ParsedAttr::AS_CXX11, "tessera_assumes"},
        {ParsedAttr::AS_CXX11, "tessera::assumes"}};
    static constexpr Spelling PreservesSpellings[] = {
        {ParsedAttr::AS_GNU, "tessera_preserves"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "tessera_preserves"},
#else
        {ParsedAttr::AS_C2x, "tessera_preserves"},
#endif
        {ParsedAttr::AS_CXX11, "tessera_preserves"},
        {ParsedAttr::AS_CXX11, "tessera::preserves"}};
    static constexpr Spelling ReadonlySpellings[] = {
        {ParsedAttr::AS_GNU, "tessera_readonly"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "tessera_readonly"},
#else
        {ParsedAttr::AS_C2x, "tessera_readonly"},
#endif
        {ParsedAttr::AS_CXX11, "tessera_readonly"},
        {ParsedAttr::AS_CXX11, "tessera::readonly"}};
    if constexpr (Kind == TesseraPropertyKind::Guarantees)
      Spellings = GuaranteesSpellings;
    else if constexpr (Kind == TesseraPropertyKind::Assumes)
      Spellings = AssumesSpellings;
    else if constexpr (Kind == TesseraPropertyKind::Preserves)
      Spellings = PreservesSpellings;
    else
      Spellings = ReadonlySpellings;
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
    return handleTesseraPropertyAttribute(S, D, Attr, Kind);
  }
};

static ParsedAttrInfoRegistry::Add<
    TesseraPropertyAttrInfo<TesseraPropertyKind::Guarantees>>
    T3("tessera_guarantees", "");
static ParsedAttrInfoRegistry::Add<
    TesseraPropertyAttrInfo<TesseraPropertyKind::Preserves>>
    T4("tessera_preserves", "");
static ParsedAttrInfoRegistry::Add<
    TesseraPropertyAttrInfo<TesseraPropertyKind::Assumes>>
    T5("tessera_assumes", "");
static ParsedAttrInfoRegistry::Add<
    TesseraPropertyAttrInfo<TesseraPropertyKind::Readonly>>
    T6("tessera_readonly", "");

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
      // Like a malformed tessera attribute, a malformed rule is left out
      // rather than failing the compile.
      unsigned ID = PP.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Warning, "'#pragma optimize' expects the rule as "
                                      "a string literal; it is ignored");
      PP.Diag(Tok.getLocation(), ID);
      PP.DiscardUntilEndOfDirective();
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
