#ifndef LLVM_TRANSFORMS_ARTS_H
#define LLVM_TRANSFORMS_ARTS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
// #include "llvm/IR/Type.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include <cstdint>

namespace llvm {

/// OMP INFO 
/// Helper Struct to get OpenMP related information
struct OMPInfo {
  enum RTFType {
    OTHER = 0,
    PARALLEL,
    PARALLEL_FOR,
    TASKALLOC,
    TASK,
    TASKWAIT,
    TASKDEP,
    SET_NUM_THREADS
  };

  /// Helper Functions
  static RTFType getRTFunction(Function *F) {
    if (!F)
      return RTFType::OTHER;
    auto CalleeName = F->getName();
    if (CalleeName == "__kmpc_fork_call")
      return RTFType::PARALLEL;
    if (CalleeName == "__kmpc_omp_task_alloc")
      return RTFType::TASKALLOC;
    if (CalleeName == "__kmpc_omp_task")
      return RTFType::TASK;
    if (CalleeName == "__kmpc_omp_task_alloc_with_deps")
      return RTFType::TASKDEP;
    if (CalleeName == "__kmpc_omp_taskwait")
      return RTFType::TASKWAIT;
    if (CalleeName == "omp_set_num_threads")
      return RTFType::SET_NUM_THREADS;
    if (CalleeName == "__kmpc_for_static_init_4")
      return RTFType::PARALLEL_FOR;
    return RTFType::OTHER;
  }

  static RTFType getRTFunction(CallBase &CB) {
    auto *Callee = CB.getCalledFunction();
    return getRTFunction(Callee);
  }

  static RTFType getRTFunction(Instruction *I) {
    auto *CB = dyn_cast<CallBase>(I);
    if (!CB)
      return RTFType::OTHER;
    return getRTFunction(*CB);
  }

  static bool isTaskFunction(Function *F) {
    auto RT = getRTFunction(F);
    if (RT == RTFType::TASK || RT == RTFType::TASKDEP ||
        RT == RTFType::TASKWAIT)
      return true;
    return false;
  }

  static bool isRTFunction(CallBase &CB) {
    auto RT = getRTFunction(CB);
    if (RT != RTFType::OTHER)
      return true;
    return false;
  }
};

/// DATA ENVIRONMENT
/// Struct to store information about the data environment of an Edt Region.
/// The value is usually a Function argument, but it can be any value.
struct DataEnv {
  enum Type { OTHER = 0, PRIVATE, SHARED, FIRSTPRIVATE, LASTPRIVATE, NONE };
  /// Interface 
  DataEnv(){};
  DataEnv(DataEnv &DE) { append(DE); };
  DataEnv &operator=(const DataEnv &DE) {
    Privates = DE.Privates;
    Shareds = DE.Shareds;
    FirstPrivates = DE.FirstPrivates;
    LastPrivates = DE.LastPrivates;
    return *this;
  }

  void append(const DataEnv &DE) {
    Privates.insert(DE.Privates.begin(), DE.Privates.end());
    Shareds.insert(DE.Shareds.begin(), DE.Shareds.end());
    FirstPrivates.insert(DE.FirstPrivates.begin(),
                            DE.FirstPrivates.end());
    LastPrivates.insert(DE.LastPrivates.begin(),
                           DE.LastPrivates.end());
  };

  /// Given a Value, it returns if it is in the Data Environment
  bool isInDE(Value *V) {
    if (Privates.count(V) || Shareds.count(V) ||
        FirstPrivates.count(V) || LastPrivates.count(V))
      return true;
    return false;
  }

  /// Insert Value to the Data Environment
  void insertValue(Value *V, Type Ty) {
    switch (Ty) {
    case Type::PRIVATE:
      Privates.insert(V);
      break;
    case Type::SHARED:
      Shareds.insert(V);
      break;
    case Type::FIRSTPRIVATE:
      FirstPrivates.insert(V);
      break;
    case Type::LASTPRIVATE:
      LastPrivates.insert(V);
      break;
    default:
      break;
    }
  }

  /// Given a Value, it returns the Type
  Type getType(Value *V) {
    if (Privates.count(V))
      return Type::PRIVATE;
    if (Shareds.count(V))
      return Type::SHARED;
    if (FirstPrivates.count(V))
      return Type::FIRSTPRIVATE;
    if (LastPrivates.count(V))
      return Type::LASTPRIVATE;
    return Type::NONE;
  }

  /// Attributes 
  SetVector<Value *> Privates;
  SetVector<Value *> Shareds;
  SetVector<Value *> FirstPrivates;
  SetVector<Value *> LastPrivates;
};

inline raw_ostream &operator<<(raw_ostream &OS, DataEnv &DE) {
  OS << "Data environment: \n";
  OS << "Firstprivate: " << DE.FirstPrivates.size() << "\n";
  for (auto *V : DE.FirstPrivates)
    OS << "  - " << *V << "\n";
  OS << "Private: " << DE.Privates.size() << "\n";
  for (auto *V : DE.Privates)
    OS << "  - " << *V << "\n";
  OS << "Shared: " << DE.Shareds.size() << "\n";
  for (auto *V : DE.Shareds)
    OS << "  - " << *V << "\n";
  OS << "Lastprivate: " << DE.LastPrivates.size() << "\n";
  for (auto *V : DE.LastPrivates)
    OS << "  - " << *V << "\n";
  return OS;
}

/// EDTs DEPENDENCIES 
/// This struct represents a dependency between two Edts
/// We may have different kind of dependencies: EVENTS or DATA DEPENDENCIES
/// DATA DEPENDENCIES: For this kind of dependencies, we need to know the
/// values that are going to be signaled to the successor Edt.
/// EVENTS: For this kind of dependencies, we need to know the event that

struct EdtInfo;
struct EdtDep {
  /// Type
  enum Type { OTHER = 0, IN, OUT, INOUT };
  /// Interface 
  EdtDep(Type Ty, EdtInfo *Edt) : Ty(Ty), EdtTo(Edt){};

  /// Attributes 
  /// Type of the dependency
  Type Ty;
  // Edt where the value will be signaled to
  EdtInfo *EdtTo;
  /// Values to be signaled
  SmallVector<Value *, 4> Values;
};

struct EventDep : public EdtDep {
  /// Interface 
  EventDep(EdtInfo *Edt, Value *Event) : EdtDep(Edt), Event(Event){};

  /// Attributes 
  /// Event to be signaled
  Value *Event;
};





/// EDT INFO 
/// Struct to store information about Edts. This is not an Edt itself,
/// but a representation of it. (E.G. A function that only accesses
/// the information of its arguments is not an Edt, but it is represented
/// by an EdtInfo object)
struct EdtInfo {
  /// Type
  enum Type{ TASK, PARALLEL, WRAPPER, OTHER, MAIN };

  /// Interface 
  EdtInfo(Type Ty, uint64_t ID) : Ty(Ty), ID(ID){};
  EdtInfo(Type Ty, uint64_t ID, Function *F) : Ty(Ty), ID(ID), F(F){};

  void setF(Function *F) { this->F = F; }
  Type getType() { return Ty; }

  /// Attributes 
  /// Edt Type
  Type Ty = Type::OTHER;
  /// GUID of the Edt
  uint64_t ID;
  /// Pointer to the Edt function
  Function *F = nullptr;
  /// Data environment of the Edt
  DataEnv DE;
  /// Indicates if the Edt transformed to an ARTS Edt or not
  bool Transformed = false;
  /// Predecessors
  SetVector<EdtInfo *> Preds;
  /// Successors
  DenseMap<EdtInfo *, EdtDep *> Succs;

  bool hasSharedVars() { return DE.Shareds.size() > 0; }

  void insertDep(EdtDep::Type Ty, EdtInfo *Edt, Value *V) {
    if (!Succs.count(Edt))
      Succs[Edt] = new EdtDep(Ty, Edt);
    Succs[Edt]->Values.push_back(V);
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, EdtInfo &EI) {
  OS << "----- Edt -----\n";
  OS << "ID: " << EI.ID << "\n";
  OS << "Function: " << EI.F->getName() << "\n";
  OS << "Type: ";
  switch (EI.Ty) {
  case EdtInfo::Type::TASK:
    OS << "TASK";
    break;
  case EdtInfo::Type::PARALLEL:
    OS << "PARALLEL";
    break;
  case EdtInfo::Type::WRAPPER:
    OS << "WRAPPER";
    break;
  case EdtInfo::Type::OTHER:
    OS << "OTHER";
    break;
  case EdtInfo::Type::MAIN:
    OS << "MAIN";
    break;
  }
  OS << "\n";
  OS << EI.DE;
  OS << "Predecessors: " << EI.Preds.size() << "\n";
  for (auto *Pred : EI.Preds)
    OS << "  - " << Pred->ID << "\n";
  OS << "Successors: " << EI.Succs.size() << "\n";
  for (auto SuccItr : EI.Succs) {
    auto *Succ = SuccItr.first;
    auto *Dep = SuccItr.second;
    OS << "  - " << Succ->ID << "\n";
    OS << "    Values: " << Dep->Values.size() << "\n";
    for (auto *V : Dep->Values)
      OS << "      - " << *V << "\n";
  }
  return OS;
}

/// ARTS TRANSFORM 
struct ARTSTransformer {
  friend struct EdtInfo;
  /// Interface 
  ARTSTransformer(Module &M) : M(M) {}
  ~ARTSTransformer() {
    /// Delete Edts
    // for (const auto &It : EdtBlocks)
    //   delete (It.second);
    // EdtBlocks.clear();
  }
  bool run(ModuleAnalysisManager &AM);
  bool runAttributor(Attributor &A);

  /// Helper Functions 

  /// Create Edt
  EdtInfo *insertEdt(EdtInfo::Type Ty, Function *F) {
    EdtInfo *Edt = new EdtInfo(Ty, EdtItr++, F);
    // EdtTypeToFunction[Ty] = F;
    EdtFunctions[F] = Edt;
    return Edt;
  }

  void insertEdtCall(CallBase *Call) {
    auto *BB = Call->getParent();
    EdtCalls[BB] = Call;
  }

  /// Get Edt
  EdtInfo *getEdt(Function *F) {
    if (EdtFunctions.count(F))
      return EdtFunctions[F];
    return nullptr;
  }

  EdtInfo *getEdt(CallBase *Call) {
    auto *F = Call->getCalledFunction();
    return getEdt(F);
  }

  EdtInfo *getEdt(BasicBlock *BB) {
    auto *Call = getEdtCall(BB);
    if (Call)
      return getEdt(Call);
    return nullptr;
  }

  /// Get the Edt that is called in a BasicBlock
  CallBase *getEdtCall(BasicBlock *BB) {
    if (EdtCalls.count(BB))
      return EdtCalls[BB];
    return nullptr;
  }

  /// Insert Type for Argument
  void insertArgToDE(Argument *Arg, DataEnv::Type Type,  DataEnv &DE) {
    DE.insertValue(Arg, Type);
  }

  void insertArgToDE(Argument *Arg, DataEnv::Type Type, EdtInfo &Edt) {
    insertArgToDE(Arg, Type, Edt.DE);
  }


  /// Attributes 
  /// The underlying module.
  Module &M;

  /// Edt Counter
  uint64_t EdtItr = 0;
  /// Maps the Function to the EdtInfo that represents it
  DenseMap<Function *, EdtInfo *> EdtFunctions;
  /// Maps the BasicBlock to the an EdtCall
  DenseMap<BasicBlock *, CallBase *> EdtCalls;
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               DenseMap<Function *, EdtInfo *> Edts) {
  OS << "Dumping Edts\n";
  OS << "Number of Edts: " << Edts.size() << "\n";
  for (auto EdtItr : Edts) {
    auto *E = EdtItr.second;
    OS << "\n" << *E;
  }
  return OS;
}

/// ARTS TRANSFORM PASS
///  From OpenMP to ARTS transformation pass.
class ARTSTransformPass : public PassInfoMixin<ARTSTransformPass> {
public:
  ARTSTransformPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_ARTS_H
