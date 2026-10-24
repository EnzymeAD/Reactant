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
#include "clang/AST/Mangle.h"
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
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"

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

// Globals whose type is that of the value behind an argument of a tessera op;
// see makeTesseraArgTypeCarriers.
static std::vector<VarDecl *> TesseraArgTypeCarriers;

// Tessera ops whose declaration must be forced into the module; see
// emitTesseraOpRefGlobals.
static std::vector<FunctionDecl *> TesseraOpFunctions;

// A fact stated on a statement, waiting for the call to its marker to be
// spliced in after that statement; see handleTesseraFactStatement.
struct TesseraFactSite {
  Stmt *stmt;
  Expr *call;
  FunctionDecl *marker;
  bool placed = false;
};

static std::vector<TesseraFactSite> TesseraFactSites;

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

/// The declaration a C++ name in a rule refers to, `KSPCG` or, qualified,
/// `mfem::IntRules` and `mfem::Geometry::CUBE`, looked up from file scope as a
/// name written there is: each qualifier is a namespace, a class or an enum.
/// Only the variable or enumerator a rule can use is returned.
static NamedDecl *lookupRuleDecl(Sema &S, StringRef name) {
  name.consume_front("::");
  SmallVector<StringRef, 4> parts;
  name.split(parts, "::");
  DeclContext *DC = S.getASTContext().getTranslationUnitDecl();
  for (size_t i = 0, e = parts.size(); i != e; ++i) {
    bool last = i + 1 == e;
    DeclarationName DN(&S.getASTContext().Idents.get(parts[i]));
    DeclContext *next = nullptr;
    for (NamedDecl *D : DC->lookup(DN)) {
      if (last && (isa<EnumConstantDecl>(D) || isa<VarDecl>(D)))
        return D;
      if (!last && (isa<NamespaceDecl>(D) || isa<CXXRecordDecl>(D) ||
                    isa<EnumDecl>(D))) {
        next = cast<DeclContext>(D);
        break;
      }
    }
    if (!next)
      return nullptr;
    DC = next;
  }
  return nullptr;
}

/// The value of the C name `name` in the syntax of a rule, if it has one that
/// a rule can hold: an enumerator becomes its value, a macro that expands to a
/// single integer or string becomes that literal, and a global variable
/// becomes its symbol, `@name`, which the rule passes the address of. A macro
/// that expands to another name is followed. A global is added to `globals`,
/// for the caller to keep in the module.
static std::optional<std::string>
resolveRuleName(Sema &S, StringRef name, SmallVectorImpl<VarDecl *> &globals,
                unsigned depth = 0) {
  if (depth > 8)
    return std::nullopt;
  Preprocessor &PP = S.getPreprocessor();
  IdentifierInfo *II =
      name.contains(':') ? nullptr : PP.getIdentifierInfo(name);

  if (const MacroInfo *MI = II ? PP.getMacroInfo(II) : nullptr) {
    if (!MI->isObjectLike() || MI->getNumTokens() != 1)
      return std::nullopt;
    const Token &tok = MI->getReplacementToken(0);
    std::string spelling = PP.getSpelling(tok);
    switch (tok.getKind()) {
    case tok::string_literal: {
      // Only a plain "..." with nothing in it a rule string cannot hold.
      StringRef text(spelling);
      if (!text.consume_front("\"") || !text.consume_back("\"") ||
          text.find_first_of("\\'") != StringRef::npos)
        return std::nullopt;
      return "'" + text.str() + "'";
    }
    case tok::numeric_constant: {
      int64_t value;
      if (StringRef(spelling).getAsInteger(0, value))
        return std::nullopt;
      return std::to_string(value);
    }
    case tok::identifier:
      return resolveRuleName(S, spelling, globals, depth + 1);
    default:
      return std::nullopt;
    }
  }

  NamedDecl *D = lookupRuleDecl(S, name);
  if (auto *ECD = dyn_cast_or_null<EnumConstantDecl>(D))
    return toString(ECD->getInitVal(), 10);
  // A global object, whose address the rule passes. A reference has no object
  // of its own to point to, and a thread-local one has no single address.
  if (auto *VD = dyn_cast_or_null<VarDecl>(D)) {
    if (!VD->hasGlobalStorage() || VD->isStaticLocal() ||
        VD->getType()->isReferenceType() ||
        VD->getTLSKind() != VarDecl::TLS_None)
      return std::nullopt;
    globals.push_back(VD);
    return "@" + ASTNameGenerator(S.getASTContext()).getName(VD);
  }
  return std::nullopt;
}

/// Every identifier in `text` that is not part of a call, a qualified C++ one
/// such as `mfem::IntRules` counting as one. A call's names, the `lib` and
/// `solve` of `lib.solve(A)` or the `SPD` of `SPD(A)`, are never a variable or
/// a value, and are left out so that a C name they happen to match is not
/// looked up.
static SmallVector<StringRef> findNonCallIdentifiers(StringRef text) {
  SmallVector<StringRef> identifiers;
  for (size_t i = 0, n = text.size(); i < n;) {
    // A run of identifier characters, with `::` joining its parts.
    size_t j = i;
    while (j < n && (isAsciiIdentifierContinue(text[j]) ||
                     text.substr(j).starts_with("::")))
      j += text[j] == ':' ? 2 : 1;
    // One starting with a digit is a number, whose letters, as the exponent
    // of `1e3`, are not an identifier.
    if (j > i && !isDigit(text[i])) {
      StringRef before = text.take_front(i).rtrim();
      StringRef after = text.drop_front(j).ltrim();
      bool partOfCall = before.ends_with(".") || after.starts_with(".") ||
                        after.starts_with("(");
      if (!partOfCall)
        identifiers.push_back(text.slice(i, j));
    }
    i = std::max(j, i + 1);
  }
  return identifiers;
}

// A global that is there only to carry annotations into the module:
//   static int <Name> __attribute__((used, annotate(...), ...)) = 0;
static void emitAnnotationCarrier(Sema &S, const Twine &name,
                                  ArrayRef<std::string> annotations) {
  auto &AST = S.getASTContext();
  SourceLocation loc;
  DeclContext *declCtx = AST.getTranslationUnitDecl();
  auto *VD =
      VarDecl::Create(AST, declCtx, loc, loc, &AST.Idents.get(name.str()),
                      AST.IntTy, nullptr, SC_Static);
  VD->setImplicit(true);
  VD->setInit(IntegerLiteral::Create(AST, llvm::APInt(32, 0), AST.IntTy, loc));
  VD->addAttr(clang::UsedAttr::CreateImplicit(AST));
  for (const std::string &annotation : annotations)
    VD->addAttr(AnnotateAttr::CreateImplicit(AST, annotation, nullptr, 0));
  declCtx->addDecl(VD);
  S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
}

/// Put each rule in the module, with the value of each C name in the rules
/// that has one: a library's header can write a rule with the library's own
/// names for its constants and globals, as `MAT_SPD`, `mfem::Geometry::CUBE`
/// or `mfem::IntRules`, and Tessera sees the rule long after those names are
/// gone. Which names a rule uses as values is for lift-tessera-annotations to
/// say, which puts the values in their place.
static void emitOptimizationRules(Sema &S, ArrayRef<std::string> rules) {
  auto &AST = S.getASTContext();
  SourceLocation loc;
  DeclContext *declCtx = AST.getTranslationUnitDecl();

  // Looked up here rather than when the pragma is read, since a rule may name
  // constants declared after it.
  llvm::SetVector<StringRef> names;
  for (const std::string &rule : rules)
    for (StringRef name : findNonCallIdentifiers(rule))
      names.insert(name);
  SmallVector<std::string> values;
  SmallVector<VarDecl *> globals;
  for (StringRef name : names)
    if (std::optional<std::string> value = resolveRuleName(S, name, globals))
      values.push_back(("tessera_name=" + name + "=" + *value).str());

  // A global a rule names is only in the module if something uses it, which
  // may be nothing but the rule:
  //   static <type> *__tessera_global_ref_N __attribute__((used)) = &global;
  llvm::SmallPtrSet<VarDecl *, 4> pinned;
  for (VarDecl *global : globals) {
    if (!pinned.insert(global).second)
      continue;
    QualType refTy = AST.getPointerType(global->getType());
    auto *DR = DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), loc, global,
                                   false, loc, global->getType(),
                                   ExprValueKind::VK_LValue, global, nullptr);
    auto *ref = UnaryOperator::Create(
        AST, DR, UnaryOperatorKind::UO_AddrOf, refTy, ExprValueKind::VK_PRValue,
        ExprObjectKind::OK_Ordinary, loc, /*CanOverflow=*/false,
        FPOptionsOverride());
    auto &Id = AST.Idents.get("__tessera_global_ref_" +
                              std::to_string(pinned.size() - 1));
    auto *VD =
        VarDecl::Create(AST, declCtx, loc, loc, &Id, refTy, nullptr, SC_Static);
    VD->setImplicit(true);
    VD->setInit(ref);
    VD->addAttr(clang::UsedAttr::CreateImplicit(AST));
    S.MarkVariableReferenced(loc, global);
    declCtx->addDecl(VD);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
  }

  for (size_t i = 0, e = rules.size(); i != e; ++i)
    emitAnnotationCarrier(S, "__tessera_optimize_rule_" + Twine(i),
                          {"tessera_optimize=" + rules[i]});
  if (!values.empty())
    emitAnnotationCarrier(S, "__tessera_names", values);
}

// Emit each argument type carrier whose type is complete by now:
//   static <type> __tessera_arg_type_N __attribute__((used)) = {};
// It is never constructed, so it is never destroyed either: it is only there
// for its type, and lift-tessera-annotations erases it.
static void emitTesseraArgTypeCarriers(Sema &S) {
  auto &AST = S.getASTContext();
  DeclContext *declCtx = AST.getTranslationUnitDecl();
  for (VarDecl *VD : TesseraArgTypeCarriers) {
    QualType type = VD->getType();
    if (!type->isObjectType() || type->isIncompleteType() ||
        type->isVariablyModifiedType())
      continue;
    if (auto *RD = type->getAsCXXRecordDecl(); RD && RD->isAbstract())
      continue;
    VD->setInit(new (AST) ImplicitValueInitExpr(type));
    VD->addAttr(clang::UsedAttr::CreateImplicit(AST));
    VD->addAttr(NoDestroyAttr::CreateImplicit(AST));
    declCtx->addDecl(VD);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
  }
}

// Emit a used global holding the function's address so that the declaration is
// created and clang emits the tessera annotation..
static void emitTesseraOpRefGlobals(Sema &S, std::vector<FunctionDecl *> &fns) {
  auto &AST = S.getASTContext();
  DeclContext *declCtx = AST.getTranslationUnitDecl();
  unsigned counter = 0;

  for (auto *FD : fns) {
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
    auto *ref = UnaryOperator::Create(
        AST, DR, UnaryOperatorKind::UO_AddrOf, refTy, ExprValueKind::VK_PRValue,
        ExprObjectKind::OK_Ordinary, loc,
        /*CanOverflow=*/false, FPOptionsOverride());

    auto &Id = AST.Idents.get("__tessera_op_ref_" + std::to_string(counter++));
    auto *VD =
        VarDecl::Create(AST, declCtx, loc, loc, &Id, refTy, nullptr, SC_Static);
    VD->setImplicit(true);
    VD->setInit(ref);
    VD->addAttr(clang::UsedAttr::CreateImplicit(AST));
    S.MarkFunctionReferenced(loc, FD);
    declCtx->addDecl(VD);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
  }
}

// Put each pending fact's marker call right after the statement it was stated
// on, anywhere below `parent`: the statement is replaced in its parent by
// `{ statement; marker(...); }`, so the call runs once the statement has.
static void spliceTesseraFacts(ASTContext &AST, Stmt *parent) {
  for (Stmt *&child : parent->children()) {
    if (!child)
      continue;
    Stmt *original = child;
    spliceTesseraFacts(AST, original);
    SmallVector<Stmt *> stmts;
    for (TesseraFactSite &site : TesseraFactSites) {
      if (site.placed || site.stmt != original)
        continue;
      if (stmts.empty())
        stmts.push_back(original);
      stmts.push_back(site.call);
      site.placed = true;
    }
    if (!stmts.empty())
      child =
          CompoundStmt::Create(AST, stmts, FPOptionsOverride(),
                               original->getBeginLoc(), original->getEndLoc());
  }
}

// Splice in the facts stated in `D`'s body, before codegen sees it: the plugin
// comes before the main action, so its consumer is handed each declaration
// first.
static void spliceTesseraFacts(ASTContext &AST, Decl *D) {
  if (llvm::all_of(TesseraFactSites,
                   [](const TesseraFactSite &site) { return site.placed; }))
    return;
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (Stmt *body = FD->getBody())
      spliceTesseraFacts(AST, body);
    return;
  }
  // A namespace or an extern "C" block reaches the consumer whole.
  if (auto *DC = dyn_cast<DeclContext>(D))
    for (Decl *inner : DC->decls())
      spliceTesseraFacts(AST, inner);
}

// Emit the marker of each fact that was placed, and say which could not be.
static void emitTesseraFactMarkers(Sema &S) {
  for (TesseraFactSite &site : TesseraFactSites) {
    if (site.placed) {
      S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(site.marker));
      continue;
    }
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "the fact stated here could not be placed in the function it is "
        "stated in; it is ignored");
    S.Diag(site.stmt->getBeginLoc(), ID);
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

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG)
      spliceTesseraFacts(CI.getASTContext(), D);
    return true;
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    spliceTesseraFacts(CI.getASTContext(), D);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Sema &S = CI.getSema();
    emitOptimizationRules(S, GlobalOptimizationRules);
    emitTesseraArgTypeCarriers(S);
    emitTesseraOpRefGlobals(S, TesseraOpFunctions);
    emitTesseraFactMarkers(S);
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

// The tessera attributes. Each passes the strings it is given on as written,
// as one annotation, "tessera_guarantees=SPD(M)" or
// "tessera_preserves=SPD,A,B", and lift-tessera-annotations, which every
// frontend shares, reads them: see there for what they say. What the plugin
// adds is only what Clang alone knows -- the names of the parameters, the
// types behind them, and the variables in scope at a statement.
//
// Only the GNU spelling __attribute__((tessera_guarantees(...))) and the
// unscoped [[tessera_guarantees(...)]] keep their arguments; the scoped
// [[tessera::guarantees(...)]] reaches the plugin with none, and is ignored
// with a warning.
enum TesseraAttrKind {
  TesseraOp,
  PureTesseraOp,
  TesseraGuarantees,
  TesseraAssumes,
  TesseraPreserves,
  TesseraReadonly
};

// Each one's GNU spelling, which also names its annotation, and its scoped
// C++ spelling.
static constexpr const char *TesseraAttrSpellings[][2] = {
    {"tessera_op", "tessera::op"},
    {"pure_tessera_op", "tessera::pure_op"},
    {"tessera_guarantees", "tessera::guarantees"},
    {"tessera_assumes", "tessera::assumes"},
    {"tessera_preserves", "tessera::preserves"},
    {"tessera_readonly", "tessera::readonly"}};

// The strings a tessera attribute is given, joined with ','. A malformed
// tessera attribute only costs what it would have given, so it is reported as
// a warning and left out, and the code compiles as though it were not there.
static std::optional<std::string>
getTesseraStrings(Sema &S, const ParsedAttr &Attr, StringRef attrName) {
  auto warn = [&](StringRef message) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Warning,
                                                     "'%0' %1; it is ignored");
    S.Diag(Attr.getLoc(), ID) << attrName << message;
  };
  if (Attr.getNumArgs() == 0) {
    warn("takes one or more strings");
    return std::nullopt;
  }
  std::string strings;
  for (unsigned i = 0, e = Attr.getNumArgs(); i != e; ++i) {
    auto *literal =
        Attr.isArgExpr(i)
            ? dyn_cast<StringLiteral>(Attr.getArgAsExpr(i)->IgnoreParenCasts())
            : nullptr;
    if (!literal) {
      warn("takes string literals");
      return std::nullopt;
    }
    if (i)
      strings += ",";
    strings += literal->getString();
  }
  return strings;
}

// The names of a function's argument positions, joined with ',', as
// lift-tessera-annotations counts them: `this` first for a member function.
// An unnamed parameter has an empty name.
static std::string getTesseraParamNames(const FunctionDecl *FD) {
  SmallVector<StringRef> names;
  const auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && MD->isInstance())
    names.push_back("this");
  for (const ParmVarDecl *param : FD->parameters())
    names.push_back(param->getName());
  return llvm::join(names, ",");
}

// For each argument position of a tessera op, `this` first, a global whose
// type is that of the value the argument could be lifted as -- what a pointer
// or reference points to, or a struct passed by value itself -- since an
// opaque pointer does not say. Returns the globals' symbols, joined with ',',
// with nothing for a position that has none. Each is emitted at the end of the
// translation unit if its type is complete by then, which a class's own is not
// while one of its members is being declared (emitTesseraArgTypeCarriers).
static std::string makeTesseraArgTypeCarriers(Sema &S, FunctionDecl *FD) {
  auto &AST = S.getASTContext();
  SmallVector<QualType> types;
  const auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && MD->isInstance())
    types.push_back(MD->getThisType()->getPointeeType());
  for (ParmVarDecl *param : FD->parameters()) {
    QualType type = param->getType();
    if (type->isPointerType() || type->isReferenceType())
      type = type->getPointeeType();
    else if (!type->isRecordType())
      type = QualType();
    types.push_back(type);
  }

  SmallVector<std::string> symbols;
  for (QualType type : types) {
    symbols.emplace_back();
    if (type.isNull() || type->isDependentType())
      continue;
    auto loc = FD->getLocation();
    auto &Id = AST.Idents.get("__tessera_arg_type_" +
                              std::to_string(TesseraArgTypeCarriers.size()));
    auto *VD = VarDecl::Create(AST, AST.getTranslationUnitDecl(), loc, loc, &Id,
                               type.getUnqualifiedType(), nullptr, SC_Static);
    VD->setImplicit(true);
    TesseraArgTypeCarriers.push_back(VD);
    symbols.back() = ASTNameGenerator(AST).getName(VD);
  }
  return llvm::join(symbols, ",");
}

// Handle facts added as annotations to a statement. The fact corresponds to
// the variable object or the object a variable in scope points to.
static ParsedAttrInfo::AttrHandling
handleTesseraFactStatement(Sema &S, Stmt *St, const ParsedAttr &Attr,
                           StringRef attrName) {
  std::optional<std::string> strings = getTesseraStrings(S, Attr, attrName);
  if (!strings)
    return ParsedAttrInfo::AttributeNotApplied;
  if (S.CurContext->isDependentContext()) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning, "a fact stated on a statement in a "
                                    "template is not supported; it is ignored");
    S.Diag(Attr.getLoc(), ID);
    return ParsedAttrInfo::AttributeNotApplied;
  }

  auto &AST = S.getASTContext();
  SourceLocation loc = Attr.getLoc();
  SmallVector<VarDecl *> vars;
  for (StringRef identifier : findNonCallIdentifiers(*strings)) {
    LookupResult R(S, DeclarationName(&AST.Idents.get(identifier)), loc,
                   Sema::LookupOrdinaryName);
    R.suppressDiagnostics();
    S.LookupName(R, S.getCurScope());
    auto *VD = R.getAsSingle<VarDecl>();
    if (!VD || llvm::is_contained(vars, VD))
      continue;
    QualType varType = VD->getType().getNonReferenceType();
    if (varType->isPointerType() || varType->isRecordType())
      vars.push_back(VD);
  }

  // void __tessera_fact_N(T0 *A, T1 *B, ...) {}
  SmallVector<QualType> paramTypes;
  SmallVector<StringRef> paramNames;
  for (VarDecl *VD : vars) {
    paramTypes.push_back(
        AST.getPointerType(VD->getType().getNonReferenceType()));
    paramNames.push_back(VD->getName());
  }
  QualType fnType = AST.getFunctionType(AST.VoidTy, paramTypes,
                                        FunctionProtoType::ExtProtoInfo());
  auto &Id = AST.Idents.get("__tessera_fact_" +
                            std::to_string(TesseraFactSites.size()));
  FunctionDecl *marker = FunctionDecl::Create(
      AST, AST.getTranslationUnitDecl(), loc, loc, DeclarationName(&Id), fnType,
      AST.getTrivialTypeSourceInfo(fnType, loc), SC_None);
  marker->setImplicit(true);
  SmallVector<ParmVarDecl *> params;
  for (auto [type, VD] : llvm::zip(paramTypes, vars))
    params.push_back(ParmVarDecl::Create(
        AST, marker, loc, loc, VD->getIdentifier(), type,
        AST.getTrivialTypeSourceInfo(type, loc), SC_None, nullptr));
  marker->setParams(params);
  marker->setBody(CompoundStmt::Create(AST, {}, FPOptionsOverride(), loc, loc));
  marker->addAttr(WeakAttr::CreateImplicit(AST));
  marker->addAttr(NoInlineAttr::CreateImplicit(AST));
  for (const std::string &annotation :
       {(attrName + "=" + *strings).str(),
        "tessera_params=" + llvm::join(paramNames, ","),
        std::string("tessera_fact")})
    marker->addAttr(AnnotateAttr::CreateImplicit(AST, annotation, nullptr, 0));

  // __tessera_fact_N(&A, ...);
  SmallVector<Expr *> args;
  for (VarDecl *VD : vars) {
    ExprResult arg = S.CreateBuiltinUnaryOp(
        loc, UO_AddrOf,
        S.BuildDeclRefExpr(VD, VD->getType().getNonReferenceType(), VK_LValue,
                           loc));
    if (arg.isInvalid())
      return ParsedAttrInfo::AttributeNotApplied;
    args.push_back(arg.get());
  }
  auto *callee = DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), loc, marker,
                                     false, loc, fnType, VK_LValue, marker);
  Expr *calleePtr = ImplicitCastExpr::Create(
      AST, AST.getPointerType(fnType), CK_FunctionToPointerDecay, callee,
      nullptr, VK_PRValue, FPOptionsOverride());
  Expr *call = CallExpr::Create(AST, calleePtr, args, AST.VoidTy, VK_PRValue,
                                loc, FPOptionsOverride());

  TesseraFactSites.push_back({St, call, marker});
  return ParsedAttrInfo::AttributeApplied;
}

template <TesseraAttrKind Kind> struct TesseraAttrInfo : public ParsedAttrInfo {
  static constexpr const char *attrName = TesseraAttrSpellings[Kind][0];
  static constexpr bool isOp = Kind == TesseraOp || Kind == PureTesseraOp;

  TesseraAttrInfo() {
    OptArgs = 15;
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, attrName},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, attrName},
#else
        {ParsedAttr::AS_C2x, attrName},
#endif
        {ParsedAttr::AS_CXX11, attrName},
        {ParsedAttr::AS_CXX11, TesseraAttrSpellings[Kind][1]}};
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
    std::optional<std::string> strings = getTesseraStrings(S, Attr, attrName);
    if (!strings)
      return AttributeNotApplied;
    auto *FD = cast<FunctionDecl>(D);
    auto &AST = S.getASTContext();
    FD->addAttr(AnnotateAttr::Create(AST,
                                     (Twine(attrName) + "=" + *strings).str(),
                                     nullptr, 0, Attr.getRange()));
    FD->addAttr(AnnotateAttr::CreateImplicit(
        AST, "tessera_params=" + getTesseraParamNames(FD), nullptr, 0));
    if (isOp) {
      // A rule's right-hand side can call the op whether or not anything
      // here does, so the function has to be in the module, along with the
      // types its arguments may be lifted as.
      FD->addAttr(AnnotateAttr::CreateImplicit(
          AST, "tessera_arg_types=" + makeTesseraArgTypeCarriers(S, FD),
          nullptr, 0));
      FD->addAttr(clang::UsedAttr::CreateImplicit(AST));
      TesseraOpFunctions.push_back(FD);
    } else if (!FD->hasAttr<AlwaysInlineAttr>() &&
               !FD->hasAttr<NoInlineAttr>()) {
      // A property is read off the call that produced a value, or off the
      // parameters of the function that assumes it, and a handle's off the
      // calls given it, so that call, or that function's own body, has to
      // survive until tessera-apply-pdl runs. LLVM inlines long before the
      // module reaches Tessera.
      FD->addAttr(NoInlineAttr::CreateImplicit(AST));
    }
    return AttributeApplied;
  }

  // Only a guarantee or an assumption can be stated on a statement; clang
  // reports the others as not applying to one.
  AttrHandling handleStmtAttribute(Sema &S, Stmt *St, const ParsedAttr &Attr,
                                   class Attr *&Result) const override {
    if constexpr (Kind == TesseraGuarantees || Kind == TesseraAssumes)
      return handleTesseraFactStatement(S, St, Attr, attrName);
    return NotHandled;
  }
};

static ParsedAttrInfoRegistry::Add<TesseraAttrInfo<TesseraOp>> T1("tessera_op",
                                                                  "");
static ParsedAttrInfoRegistry::Add<TesseraAttrInfo<PureTesseraOp>>
    T2("pure_tessera_op", "");
static ParsedAttrInfoRegistry::Add<TesseraAttrInfo<TesseraGuarantees>>
    T3("tessera_guarantees", "");
static ParsedAttrInfoRegistry::Add<TesseraAttrInfo<TesseraAssumes>>
    T4("tessera_assumes", "");
static ParsedAttrInfoRegistry::Add<TesseraAttrInfo<TesseraPreserves>>
    T5("tessera_preserves", "");
static ParsedAttrInfoRegistry::Add<TesseraAttrInfo<TesseraReadonly>>
    T6("tessera_readonly", "");

struct PerfifyOpAttrInfo : public ParsedAttrInfo {
  PerfifyOpAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__((perfify_op("..."))) and C++/C2x-style
    // [[perfify_op("...")]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "perfify_op"},
#if LLVM_VERSION_MAJOR > 17
        {ParsedAttr::AS_C23, "perfify_op"},
#else
        {ParsedAttr::AS_C2x, "perfify_op"},
#endif
        {ParsedAttr::AS_CXX11, "perfify_op"},
        {ParsedAttr::AS_CXX11, "perfify::op"}};
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
    if (Attr.getNumArgs() == 0 && Attr.isCXX11Attribute() &&
        Attr.hasScope()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "the scoped C++11 spelling [[perfify::op]] cannot carry an "
          "argument list; write [[perfify_op(\"...\")]] or "
          "__attribute__((perfify_op(\"...\"))) instead");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    if (Attr.getNumArgs() != 1) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'perfify_op' attribute requires a single string argument");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    auto *Arg0 = Attr.getArgAsExpr(0);
    StringLiteral *Literal = dyn_cast<StringLiteral>(Arg0->IgnoreParenCasts());
    if (!Literal) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "argument to 'perfify_op' "
                                    "attribute must be a string literal");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    D->addAttr(AnnotateAttr::Create(
        S.Context, ("perfify_op=" + Literal->getString()).str(), nullptr, 0,
        Attr.getRange()));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<PerfifyOpAttrInfo>
    PerfifyOpX("perfify_op", "");

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

    std::string optimizationRule;
    if (!PP.FinishLexStringLiteral(Tok, optimizationRule, "pragma optimize",
                                   /*AllowMacroExpansion=*/false))
      return;
    if (Tok.isNot(tok::eod)) {
      PP.Diag(Tok, diag::ext_pp_extra_tokens_at_eol)
          << "pragma optimize warning";
      return;
    }
    GlobalOptimizationRules.push_back(optimizationRule);
  }
};

static PragmaHandlerRegistry::Add<PragmaTesseraOptimizeHandler>
    OptX("optimize", "custom tessera optimization");
} // namespace

#endif
// timing
