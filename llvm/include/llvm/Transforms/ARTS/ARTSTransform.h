#ifndef LLVM_TRANSFORMS_ARTS_H
#define LLVM_TRANSFORMS_ARTS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
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

/// EDT DEP 
struct EdtInfo;
struct EdtDep {
  /// Interface 
  EdtDep() : Edt(nullptr){};
  EdtDep(EdtInfo *Edt) : Edt(Edt){};
  // EdtDep(EdtInfo *Edt, DataEnv *DE) : Edt(Edt), DE(DE) {};

  /// Attributes 
  // Edt where the value will be signaled to
  EdtInfo *Edt;
  /// Values to be signaled
  SmallVector<Value *, 4> Values;
};

/// EDT INFO 
/// Struct to store information about Edts. This is not an Edt itself,
/// but a representation of it. (E.G. A function that only accesses
/// the information of its arguments is not an Edt, but it is represented
/// by an EdtInfo object)
struct EdtInfo {
  /// Type
  enum Type{ TASK, PARALLEL, WRAPPER, OTHER };

  /// Interface 
  EdtInfo(Type Ty, uint64_t ID) : Ty(Ty), ID(ID){};
  EdtInfo(Type Ty, uint64_t ID, Function *F) : ID(ID), F(F){};

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
  }
  OS << "\n";
  OS << EI.DE;
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
  bool run(FunctionAnalysisManager &FAM);
  bool runAttributor(Attributor &A);

  /// Helper Functions 

  /// Create Edt
  EdtInfo *insertEdt(EdtInfo::Type Ty, Function *F) {
    EdtInfo *Edt = new EdtInfo(Ty, EdtItr++, F);
    EdtTypeToFunction[Ty] = F;
    EdtForFunction[F] = Edt;
    return Edt;
  }

  /// Get Edt
  EdtInfo *getEdt(Function *F) {
    if (EdtForFunction.count(F))
      return EdtForFunction[F];
    return nullptr;
  }

  /// Insert Type for Argument
  void insertArgToDE(Argument *Arg, DataEnv::Type Type,  DataEnv &DE) {
    DE.insertValue(Arg, Type);
    TypeForArg[Arg] = Type;
  }

  void insertArgToDE(Argument *Arg, DataEnv::Type Type, EdtInfo &Edt) {
    insertArgToDE(Arg, Type, Edt.DE);
  }

  /// Get Argument Type
  DataEnv::Type getType(Argument *Arg) {
    if (TypeForArg.count(Arg))
      return TypeForArg[Arg];
    return DataEnv::Type::NONE;
  }
  

  /// Attributes 
  /// The underlying module.
  Module &M;

  /// Edt Counter
  uint64_t EdtItr = 0;
  /// Maps the EdtType to the function
  DenseMap<uint8_t, Function *> EdtTypeToFunction;
  /// Maps the Function to the EdtInfo that represents it
  DenseMap<Function *, EdtInfo *> EdtForFunction;
  /// Maps a Function Argument to a DataEnvironment Type
  DenseMap<Argument *, DataEnv::Type> TypeForArg;
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               SmallPtrSet<EdtInfo *, 4> &Edts) {
  OS << "DUMPING Edts\n";
  // OS << "Number of Edts: " << Edts.size() << "\n";
  // for (auto *E : Edts) {
  //   OS << "\n" << *E;
  // }
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
