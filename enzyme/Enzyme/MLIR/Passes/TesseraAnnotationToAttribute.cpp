#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "Passes/Passes.h"

namespace mlir {
namespace enzyme {

class TesseraAnnotationToAttributePass
    : public PassWrapper<TesseraAnnotationToAttributePass, OperationPass<ModuleOp>> {

public:
  StringRef getArgument() const final { return "tessera-annotation-to-attribute"; }
  StringRef getDescription() const final {
    return "Convert LLVM global annotations to function attributes for tessera ops";
  }

  void runOnOperation() override {
  llvm::errs() << "=== TesseraAnnotationToAttribute pass is running! ===\n";
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  // Step 1: Collect string constants
  DenseMap<StringRef, std::string> stringGlobals;
  for (auto global : module.getOps<LLVM::GlobalOp>()) {
    if (global.getSection() && *global.getSection() == "llvm.metadata") {
      if (auto strAttr = global.getValueAttr().dyn_cast_or_null<StringAttr>()) {
        stringGlobals[global.getSymName()] = strAttr.getValue().str();
      }
    }
  }

  // Step 2: Find llvm.global.annotations
  LLVM::GlobalOp annotationGlobal = nullptr;
  for (auto global : module.getOps<LLVM::GlobalOp>()) {
    if (global.getSymName() == "llvm.global.annotations") {
      annotationGlobal = global;
      break;
    }
  }

  if (!annotationGlobal)
    return;

  Region &region = annotationGlobal.getInitializerRegion();
  if (region.empty())
    return;

  // Step 3: Track what each SSA value represents
  // We need to know: which values are function pointers, which are annotation strings
  DenseMap<Value, StringRef> valueToFunction;  // Maps SSA value -> function name
  DenseMap<Value, StringRef> valueToAnnotation; // Maps SSA value -> annotation string

  // First pass: find all addressof operations
  for (Operation &op : region.front()) {
    if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(&op)) {
      StringRef globalName = addrOf.getGlobalName();
      Value result = addrOf.getResult();

      // Is this a function?
      if (module.lookupSymbol<LLVM::LLVMFuncOp>(globalName)) {
        valueToFunction[result] = globalName;
        llvm::errs() << "Found function address: " << globalName << "\n";
      }
      // Is this a string constant?
      else if (stringGlobals.count(globalName)) {
        valueToAnnotation[result] = stringGlobals[globalName];
        llvm::errs() << "Found annotation string: " << stringGlobals[globalName] << "\n";
      }
    }
  }

  // Step 4: Track insertvalue operations to connect functions with their annotations
  // Each annotation struct has the function at position [0] and annotation at position [1]
  DenseMap<Value, StringRef> structToFunction;   // Tracks which function a struct describes
  DenseMap<Value, StringRef> structToAnnotation; // Tracks which annotation a struct has

  for (Operation &op : region.front()) {
    if (auto insertValue = dyn_cast<LLVM::InsertValueOp>(&op)) {
      Value inserted = insertValue.getValue();
      Value container = insertValue.getContainer();
      Value result = insertValue.getResult();
      auto position = insertValue.getPosition();

      if (position.size() == 1) {
        // Inserting into the struct itself
        if (position[0] == 0 && valueToFunction.count(inserted)) {
          // Position 0 is the function pointer
          structToFunction[result] = valueToFunction[inserted];
          llvm::errs() << "Struct at position 0 gets function: "
                      << valueToFunction[inserted] << "\n";
        } else if (position[0] == 1 && valueToAnnotation.count(inserted)) {
          // Position 1 is the annotation string
          structToAnnotation[result] = valueToAnnotation[inserted];
          llvm::errs() << "Struct at position 1 gets annotation: "
                      << valueToAnnotation[inserted] << "\n";
        }

        // Propagate information through the chain
        if (structToFunction.count(container)) {
          structToFunction[result] = structToFunction[container];
        }
        if (structToAnnotation.count(container)) {
          structToAnnotation[result] = structToAnnotation[container];
        }
      }
    }
  }

  // Step 5: Find complete annotation entries (struct with both function and annotation)
  DenseMap<StringRef, StringRef> functionToAnnotation;

  for (auto [structValue, funcName] : structToFunction) {
    if (structToAnnotation.count(structValue)) {
      StringRef annotStr = structToAnnotation[structValue];
      functionToAnnotation[funcName] = annotStr;
      llvm::errs() << "Complete annotation: " << funcName
                  << " -> " << annotStr << "\n";
    }
  }

  // Step 6: Apply attributes to the correct functions only
  for (auto [funcName, annotStr] : functionToAnnotation) {
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!func)
      continue;

    // Parse "tessera_op=reciprocal\0"
    StringRef annot(annotStr);
    if (annot.starts_with("tessera_op=")) {
      StringRef opName = annot.substr(11);
      opName = opName.take_while([](char c) { return c != '\0'; });

      llvm::errs() << "Applying tessera.op=\"" << opName
                  << "\" to function " << funcName << "\n";
      func->setAttr("tessera.op", builder.getStringAttr(opName));
    }
  }

  // Step 7: Clean up
  annotationGlobal.erase();
  SmallVector<LLVM::GlobalOp> toErase;
  for (auto global : module.getOps<LLVM::GlobalOp>()) {
    if (global.getSection() && *global.getSection() == "llvm.metadata") {
      toErase.push_back(global);
    }
  }
  for (auto global : toErase) {
    global.erase();
  }
}
};

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createTesseraAnnotationToAttributePass() {
  return std::make_unique<TesseraAnnotationToAttributePass>();
}
} // namespace enzyme
} // namespace mlir

