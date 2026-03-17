//! LLVM IR representation types for the slicer.
//!
//! Provides an in-memory representation of LLVM IR sufficient for
//! program slicing and dataflow analysis of TLS/SSH protocol code.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use serde::{Serialize, Deserialize};
use indexmap::IndexMap;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// LLVM IR type representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    Void,
    Int(u32),
    Float,
    Double,
    Pointer(Box<Type>),
    Array(Box<Type>, usize),
    Struct(Vec<Type>),
    NamedStruct(String),
    Function(Box<Type>, Vec<Type>, bool), // ret, params, is_vararg
    Vector(Box<Type>, usize),
    Opaque(String),
    Label,
    Metadata,
    Token,
}

impl Type {
    pub fn i1() -> Self { Type::Int(1) }
    pub fn i8() -> Self { Type::Int(8) }
    pub fn i16() -> Self { Type::Int(16) }
    pub fn i32() -> Self { Type::Int(32) }
    pub fn i64() -> Self { Type::Int(64) }
    pub fn ptr(inner: Type) -> Self { Type::Pointer(Box::new(inner)) }
    pub fn array(elem: Type, len: usize) -> Self { Type::Array(Box::new(elem), len) }
    pub fn func(ret: Type, params: Vec<Type>) -> Self {
        Type::Function(Box::new(ret), params, false)
    }
    pub fn vararg_func(ret: Type, params: Vec<Type>) -> Self {
        Type::Function(Box::new(ret), params, true)
    }

    /// Return size in bits, or None for unsized types.
    pub fn size_bits(&self) -> Option<u64> {
        match self {
            Type::Void => Some(0),
            Type::Int(w) => Some(*w as u64),
            Type::Float => Some(32),
            Type::Double => Some(64),
            Type::Pointer(_) => Some(64),
            Type::Array(elem, len) => elem.size_bits().map(|s| s * (*len as u64)),
            Type::Struct(fields) => {
                let mut total = 0u64;
                for f in fields {
                    total += f.size_bits()?;
                }
                Some(total)
            }
            Type::Vector(elem, len) => elem.size_bits().map(|s| s * (*len as u64)),
            _ => None,
        }
    }

    /// Whether this is a pointer type.
    pub fn is_pointer(&self) -> bool { matches!(self, Type::Pointer(_)) }

    /// Whether this is an integer type.
    pub fn is_integer(&self) -> bool { matches!(self, Type::Int(_)) }

    /// Whether this is a floating-point type.
    pub fn is_float(&self) -> bool { matches!(self, Type::Float | Type::Double) }

    /// Whether this is a function type.
    pub fn is_function(&self) -> bool { matches!(self, Type::Function(..)) }

    /// Get the element type if this is a pointer or array.
    pub fn element_type(&self) -> Option<&Type> {
        match self {
            Type::Pointer(inner) | Type::Array(inner, _) | Type::Vector(inner, _) => Some(inner),
            _ => None,
        }
    }

    /// Get return type if this is a function type.
    pub fn return_type(&self) -> Option<&Type> {
        match self {
            Type::Function(ret, _, _) => Some(ret),
            _ => None,
        }
    }

    /// Get parameter types if this is a function type.
    pub fn param_types(&self) -> Option<&[Type]> {
        match self {
            Type::Function(_, params, _) => Some(params),
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int(w) => write!(f, "i{}", w),
            Type::Float => write!(f, "float"),
            Type::Double => write!(f, "double"),
            Type::Pointer(inner) => write!(f, "{}*", inner),
            Type::Array(elem, len) => write!(f, "[{} x {}]", len, elem),
            Type::Struct(fields) => {
                write!(f, "{{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", field)?;
                }
                write!(f, "}}")
            }
            Type::NamedStruct(name) => write!(f, "%{}", name),
            Type::Function(ret, params, va) => {
                write!(f, "{} (", ret)?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                if *va { write!(f, ", ...")?; }
                write!(f, ")")
            }
            Type::Vector(elem, len) => write!(f, "<{} x {}>", len, elem),
            Type::Opaque(name) => write!(f, "opaque({})", name),
            Type::Label => write!(f, "label"),
            Type::Metadata => write!(f, "metadata"),
            Type::Token => write!(f, "token"),
        }
    }
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// An SSA value in the IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Named local register: %name
    Register(String, Type),
    /// Integer constant
    IntConst(i64, Type),
    /// Float constant
    FloatConst(f64, Type),
    /// Null pointer
    NullPtr(Type),
    /// Undefined value
    Undef(Type),
    /// Zero initializer
    ZeroInit(Type),
    /// Global reference: @name
    GlobalRef(String, Type),
    /// Function reference
    FunctionRef(String, Type),
    /// String constant
    StringConst(String),
    /// Block label
    BlockLabel(String),
    /// Inline constant expression
    ConstExpr(Box<Instruction>),
    /// Aggregate / vector constant
    Aggregate(Vec<Value>, Type),
    /// Poison value
    Poison(Type),
}

impl Value {
    pub fn reg(name: impl Into<String>, ty: Type) -> Self {
        Value::Register(name.into(), ty)
    }
    pub fn int(val: i64, bits: u32) -> Self {
        Value::IntConst(val, Type::Int(bits))
    }
    pub fn global(name: impl Into<String>, ty: Type) -> Self {
        Value::GlobalRef(name.into(), ty)
    }
    pub fn func_ref(name: impl Into<String>, ty: Type) -> Self {
        Value::FunctionRef(name.into(), ty)
    }
    pub fn null(inner: Type) -> Self {
        Value::NullPtr(Type::Pointer(Box::new(inner)))
    }

    /// Get the type of this value. Returns an owned Type since some variants
    /// need to construct the type dynamically.
    pub fn get_type(&self) -> Type {
        match self {
            Value::Register(_, t) | Value::IntConst(_, t) | Value::FloatConst(_, t)
            | Value::NullPtr(t) | Value::Undef(t) | Value::ZeroInit(t)
            | Value::GlobalRef(_, t) | Value::FunctionRef(_, t)
            | Value::Aggregate(_, t) | Value::Poison(t) => t.clone(),
            Value::StringConst(_) => Type::Pointer(Box::new(Type::Int(8))),
            Value::BlockLabel(_) => Type::Label,
            Value::ConstExpr(instr) => {
                instr.result_type().cloned().unwrap_or(Type::Void)
            }
        }
    }

    /// Get a reference to the type if directly stored. For variants that
    /// construct their type dynamically, use `get_type()` instead.
    pub fn ty(&self) -> &Type {
        match self {
            Value::Register(_, t) | Value::IntConst(_, t) | Value::FloatConst(_, t)
            | Value::NullPtr(t) | Value::Undef(t) | Value::ZeroInit(t)
            | Value::GlobalRef(_, t) | Value::FunctionRef(_, t)
            | Value::Aggregate(_, t) | Value::Poison(t) => t,
            // For variants without a stored type, we return a placeholder.
            // Callers needing the real type should use get_type().
            _ => {
                static VOID: Type = Type::Void;
                &VOID
            }
        }
    }

    /// Get the name if this is a register or global.
    pub fn name(&self) -> Option<&str> {
        match self {
            Value::Register(n, _) | Value::GlobalRef(n, _) | Value::FunctionRef(n, _) => Some(n),
            _ => None,
        }
    }

    /// Whether this is a constant.
    pub fn is_constant(&self) -> bool {
        matches!(self,
            Value::IntConst(..) | Value::FloatConst(..) | Value::NullPtr(_)
            | Value::Undef(_) | Value::ZeroInit(_) | Value::StringConst(_)
            | Value::Aggregate(..) | Value::Poison(_)
        )
    }

    /// Whether this references a function.
    pub fn is_function_ref(&self) -> bool {
        matches!(self, Value::FunctionRef(..))
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Register(n, _) => write!(f, "%{}", n),
            Value::IntConst(v, t) => write!(f, "{} {}", t, v),
            Value::FloatConst(v, t) => write!(f, "{} {}", t, v),
            Value::NullPtr(t) => write!(f, "{} null", t),
            Value::Undef(t) => write!(f, "{} undef", t),
            Value::ZeroInit(t) => write!(f, "{} zeroinitializer", t),
            Value::GlobalRef(n, _) => write!(f, "@{}", n),
            Value::FunctionRef(n, _) => write!(f, "@{}", n),
            Value::StringConst(s) => write!(f, "c\"{}\"", s),
            Value::BlockLabel(l) => write!(f, "label %{}", l),
            Value::ConstExpr(instr) => write!(f, "constexpr({})", instr),
            Value::Aggregate(vals, _) => {
                write!(f, "{{ ")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, " }}")
            }
            Value::Poison(t) => write!(f, "{} poison", t),
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison / Binary Operators
// ---------------------------------------------------------------------------

/// Integer comparison predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntPredicate {
    Eq, Ne, Ugt, Uge, Ult, Ule, Sgt, Sge, Slt, Sle,
}

/// Float comparison predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatPredicate {
    OEq, OGt, OGe, OLt, OLe, ONe, Ord, Uno, UEq, UGt, UGe, ULt, ULe, UNe, True, False,
}

/// Binary operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add, Sub, Mul, UDiv, SDiv, URem, SRem,
    Shl, LShr, AShr, And, Or, Xor,
    FAdd, FSub, FMul, FDiv, FRem,
}

/// Cast operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CastOp {
    Trunc, ZExt, SExt, FPToUI, FPToSI, UIToFP, SIToFP,
    FPTrunc, FPExt, PtrToInt, IntToPtr, BitCast, AddrSpaceCast,
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

/// A single LLVM IR instruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Instruction {
    /// %dest = alloca <ty>
    Alloca {
        dest: String,
        ty: Type,
        num_elements: Option<Value>,
        align: Option<u32>,
    },
    /// %dest = load <ty>, <ty>* %ptr
    Load {
        dest: String,
        ty: Type,
        ptr: Value,
        volatile: bool,
        align: Option<u32>,
    },
    /// store <ty> %val, <ty>* %ptr
    Store {
        value: Value,
        ptr: Value,
        volatile: bool,
        align: Option<u32>,
    },
    /// %dest = getelementptr <ty>, <ty>* %ptr, indices...
    GetElementPtr {
        dest: String,
        base_ty: Type,
        ptr: Value,
        indices: Vec<Value>,
        inbounds: bool,
    },
    /// %dest = call <ty> @func(args...)
    Call {
        dest: Option<String>,
        func: Value,
        args: Vec<Value>,
        ret_ty: Type,
        is_tail: bool,
        calling_conv: Option<String>,
        attrs: Vec<String>,
    },
    /// %dest = invoke <ty> @func(args...) to label %normal unwind label %unwind
    Invoke {
        dest: Option<String>,
        func: Value,
        args: Vec<Value>,
        ret_ty: Type,
        normal_dest: String,
        unwind_dest: String,
    },
    /// br label %dest
    Br { dest: String },
    /// br i1 %cond, label %true_dest, label %false_dest
    CondBr {
        cond: Value,
        true_dest: String,
        false_dest: String,
    },
    /// switch <ty> %val, label %default [ <ty> <val>, label %dest ... ]
    Switch {
        value: Value,
        default_dest: String,
        cases: Vec<(Value, String)>,
    },
    /// ret <ty> %val | ret void
    Ret { value: Option<Value> },
    /// %dest = phi <ty> [ %val1, %bb1 ], [ %val2, %bb2 ], ...
    Phi {
        dest: String,
        ty: Type,
        incoming: Vec<(Value, String)>,
    },
    /// %dest = select i1 %cond, <ty> %true_val, <ty> %false_val
    Select {
        dest: String,
        cond: Value,
        true_val: Value,
        false_val: Value,
    },
    /// %dest = <binop> <ty> %lhs, %rhs
    BinaryOp {
        dest: String,
        op: BinOp,
        lhs: Value,
        rhs: Value,
    },
    /// %dest = icmp <pred> <ty> %lhs, %rhs
    ICmp {
        dest: String,
        pred: IntPredicate,
        lhs: Value,
        rhs: Value,
    },
    /// %dest = fcmp <pred> <ty> %lhs, %rhs
    FCmp {
        dest: String,
        pred: FloatPredicate,
        lhs: Value,
        rhs: Value,
    },
    /// %dest = <castop> <ty> %val to <ty2>
    Cast {
        dest: String,
        op: CastOp,
        value: Value,
        to_ty: Type,
    },
    /// %dest = extractvalue <ty> %val, indices...
    ExtractValue {
        dest: String,
        aggregate: Value,
        indices: Vec<u32>,
    },
    /// %dest = insertvalue <ty> %agg, <ty> %val, indices...
    InsertValue {
        dest: String,
        aggregate: Value,
        value: Value,
        indices: Vec<u32>,
    },
    /// landingpad
    LandingPad {
        dest: String,
        ty: Type,
        is_cleanup: bool,
        clauses: Vec<Value>,
    },
    /// resume <ty> %val
    Resume { value: Value },
    /// unreachable
    Unreachable,
    /// %dest = atomicrmw <op> <ty>* %ptr, <ty> %val
    AtomicRMW {
        dest: String,
        op: String,
        ptr: Value,
        value: Value,
        ordering: String,
    },
    /// %dest = cmpxchg <ty>* %ptr, <ty> %cmp, <ty> %new
    CmpXchg {
        dest: String,
        ptr: Value,
        cmp: Value,
        new_val: Value,
        success_ordering: String,
        failure_ordering: String,
    },
    /// fence
    Fence { ordering: String },
    /// freeze
    Freeze { dest: String, value: Value },
    /// %dest = va_arg <ty>* %list, <ty>
    VaArg { dest: String, list: Value, ty: Type },
    /// Inline assembly
    InlineAsm {
        dest: Option<String>,
        asm_string: String,
        constraints: String,
        args: Vec<Value>,
        side_effects: bool,
    },
    /// Debug / metadata intrinsics — we keep them for source mapping.
    DebugDeclare {
        variable: String,
        value: Value,
        expression: String,
    },
    /// Memcpy/memmove/memset intrinsics
    MemIntrinsic {
        kind: MemIntrinsicKind,
        dest_ptr: Value,
        src_or_val: Value,
        length: Value,
        volatile: bool,
    },
}

/// Kind of memory intrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemIntrinsicKind {
    Memcpy,
    Memmove,
    Memset,
}

impl Instruction {
    /// Get the destination register name, if this instruction defines one.
    pub fn dest(&self) -> Option<&str> {
        match self {
            Instruction::Alloca { dest, .. }
            | Instruction::Load { dest, .. }
            | Instruction::GetElementPtr { dest, .. }
            | Instruction::BinaryOp { dest, .. }
            | Instruction::ICmp { dest, .. }
            | Instruction::FCmp { dest, .. }
            | Instruction::Cast { dest, .. }
            | Instruction::Phi { dest, .. }
            | Instruction::Select { dest, .. }
            | Instruction::ExtractValue { dest, .. }
            | Instruction::InsertValue { dest, .. }
            | Instruction::LandingPad { dest, .. }
            | Instruction::AtomicRMW { dest, .. }
            | Instruction::CmpXchg { dest, .. }
            | Instruction::Freeze { dest, .. }
            | Instruction::VaArg { dest, .. } => Some(dest),
            Instruction::Call { dest, .. } | Instruction::Invoke { dest, .. }
            | Instruction::InlineAsm { dest, .. } => dest.as_deref(),
            _ => None,
        }
    }

    /// Get all values used (read) by this instruction.
    pub fn operands(&self) -> Vec<&Value> {
        match self {
            Instruction::Alloca { num_elements, .. } => {
                num_elements.iter().collect()
            }
            Instruction::Load { ptr, .. } => vec![ptr],
            Instruction::Store { value, ptr, .. } => vec![value, ptr],
            Instruction::GetElementPtr { ptr, indices, .. } => {
                let mut ops = vec![ptr];
                ops.extend(indices.iter());
                ops
            }
            Instruction::Call { func, args, .. } | Instruction::Invoke { func, args, .. } => {
                let mut ops = vec![func];
                ops.extend(args.iter());
                ops
            }
            Instruction::Br { .. } => vec![],
            Instruction::CondBr { cond, .. } => vec![cond],
            Instruction::Switch { value, .. } => vec![value],
            Instruction::Ret { value } => value.iter().collect(),
            Instruction::Phi { incoming, .. } => {
                incoming.iter().map(|(v, _)| v).collect()
            }
            Instruction::Select { cond, true_val, false_val, .. } => {
                vec![cond, true_val, false_val]
            }
            Instruction::BinaryOp { lhs, rhs, .. } => vec![lhs, rhs],
            Instruction::ICmp { lhs, rhs, .. } => vec![lhs, rhs],
            Instruction::FCmp { lhs, rhs, .. } => vec![lhs, rhs],
            Instruction::Cast { value, .. } => vec![value],
            Instruction::ExtractValue { aggregate, .. } => vec![aggregate],
            Instruction::InsertValue { aggregate, value, .. } => vec![aggregate, value],
            Instruction::LandingPad { clauses, .. } => clauses.iter().collect(),
            Instruction::Resume { value } => vec![value],
            Instruction::Unreachable => vec![],
            Instruction::AtomicRMW { ptr, value, .. } => vec![ptr, value],
            Instruction::CmpXchg { ptr, cmp, new_val, .. } => vec![ptr, cmp, new_val],
            Instruction::Fence { .. } => vec![],
            Instruction::Freeze { value, .. } => vec![value],
            Instruction::VaArg { list, .. } => vec![list],
            Instruction::InlineAsm { args, .. } => args.iter().collect(),
            Instruction::DebugDeclare { value, .. } => vec![value],
            Instruction::MemIntrinsic { dest_ptr, src_or_val, length, .. } => {
                vec![dest_ptr, src_or_val, length]
            }
        }
    }

    /// Get register names used by this instruction.
    pub fn used_registers(&self) -> Vec<&str> {
        self.operands()
            .into_iter()
            .filter_map(|v| match v {
                Value::Register(name, _) => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Whether this instruction is a terminator.
    pub fn is_terminator(&self) -> bool {
        matches!(self,
            Instruction::Br { .. } | Instruction::CondBr { .. }
            | Instruction::Switch { .. } | Instruction::Ret { .. }
            | Instruction::Invoke { .. } | Instruction::Resume { .. }
            | Instruction::Unreachable
        )
    }

    /// Whether this instruction accesses memory.
    pub fn accesses_memory(&self) -> bool {
        matches!(self,
            Instruction::Load { .. } | Instruction::Store { .. }
            | Instruction::Call { .. } | Instruction::Invoke { .. }
            | Instruction::AtomicRMW { .. } | Instruction::CmpXchg { .. }
            | Instruction::MemIntrinsic { .. }
        )
    }

    /// Whether this is a call or invoke instruction.
    pub fn is_call(&self) -> bool {
        matches!(self, Instruction::Call { .. } | Instruction::Invoke { .. })
    }

    /// Get the called function name if this is a direct call.
    pub fn called_function_name(&self) -> Option<&str> {
        match self {
            Instruction::Call { func, .. } | Instruction::Invoke { func, .. } => {
                match func {
                    Value::FunctionRef(name, _) | Value::GlobalRef(name, _) => Some(name),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Get the result type of this instruction.
    pub fn result_type(&self) -> Option<&Type> {
        match self {
            Instruction::Alloca { ty, .. } => Some(ty),
            Instruction::Load { ty, .. } => Some(ty),
            Instruction::GetElementPtr { base_ty, .. } => Some(base_ty),
            Instruction::Call { ret_ty, .. } | Instruction::Invoke { ret_ty, .. } => {
                if *ret_ty != Type::Void { Some(ret_ty) } else { None }
            }
            Instruction::BinaryOp { lhs, .. } => Some(lhs.ty()),
            Instruction::ICmp { .. } | Instruction::FCmp { .. } => {
                static I1: Type = Type::Int(1);
                Some(&I1)
            }
            Instruction::Cast { to_ty, .. } => Some(to_ty),
            Instruction::Phi { ty, .. } => Some(ty),
            Instruction::Select { true_val, .. } => Some(true_val.ty()),
            _ => None,
        }
    }

    /// Get successor block labels for terminator instructions.
    pub fn successor_labels(&self) -> Vec<&str> {
        match self {
            Instruction::Br { dest } => vec![dest.as_str()],
            Instruction::CondBr { true_dest, false_dest, .. } => {
                vec![true_dest.as_str(), false_dest.as_str()]
            }
            Instruction::Switch { default_dest, cases, .. } => {
                let mut succs = vec![default_dest.as_str()];
                for (_, label) in cases {
                    succs.push(label.as_str());
                }
                succs
            }
            Instruction::Invoke { normal_dest, unwind_dest, .. } => {
                vec![normal_dest.as_str(), unwind_dest.as_str()]
            }
            _ => vec![],
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Load { dest, ty, ptr, .. } => {
                write!(f, "%{} = load {}, {} {}", dest, ty, ty, ptr)
            }
            Instruction::Store { value, ptr, .. } => {
                write!(f, "store {} {}, {}* {}", value.ty(), value, value.ty(), ptr)
            }
            Instruction::Call { dest, func, args, ret_ty, .. } => {
                if let Some(d) = dest {
                    write!(f, "%{} = call {} {}(", d, ret_ty, func)?;
                } else {
                    write!(f, "call {} {}(", ret_ty, func)?;
                }
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
            Instruction::Br { dest } => write!(f, "br label %{}", dest),
            Instruction::CondBr { cond, true_dest, false_dest } => {
                write!(f, "br i1 {}, label %{}, label %{}", cond, true_dest, false_dest)
            }
            Instruction::Ret { value: Some(v) } => write!(f, "ret {}", v),
            Instruction::Ret { value: None } => write!(f, "ret void"),
            Instruction::Phi { dest, ty, incoming } => {
                write!(f, "%{} = phi {} ", dest, ty)?;
                for (i, (val, bb)) in incoming.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "[{}, %{}]", val, bb)?;
                }
                Ok(())
            }
            Instruction::BinaryOp { dest, op, lhs, rhs } => {
                write!(f, "%{} = {:?} {}, {}", dest, op, lhs, rhs)
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

// ---------------------------------------------------------------------------
// Basic Block
// ---------------------------------------------------------------------------

/// A basic block containing a sequence of instructions ending with a terminator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    pub name: String,
    pub instructions: Vec<Instruction>,
    pub predecessors: Vec<String>,
    pub successors: Vec<String>,
}

impl BasicBlock {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }

    /// Push an instruction. If it's a terminator, update successors.
    pub fn push(&mut self, instr: Instruction) {
        let succs: Vec<String> = instr.successor_labels()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        if !succs.is_empty() {
            self.successors = succs;
        }
        self.instructions.push(instr);
    }

    /// Get the terminator instruction (last instruction if it's a terminator).
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instructions.last().filter(|i| i.is_terminator())
    }

    /// Get all non-terminator instructions.
    pub fn body(&self) -> &[Instruction] {
        if self.instructions.last().map_or(false, |i| i.is_terminator()) {
            &self.instructions[..self.instructions.len() - 1]
        } else {
            &self.instructions
        }
    }

    /// Names of all registers defined in this block.
    pub fn defs(&self) -> Vec<&str> {
        self.instructions.iter().filter_map(|i| i.dest()).collect()
    }

    /// Names of all registers used in this block.
    pub fn uses(&self) -> Vec<&str> {
        self.instructions.iter().flat_map(|i| i.used_registers()).collect()
    }

    /// Whether this block is empty (no instructions).
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }
}

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

/// Function attributes relevant to protocol analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FunctionAttribute {
    NoReturn,
    NoUnwind,
    ReadOnly,
    ReadNone,
    WriteOnly,
    AlwaysInline,
    NoInline,
    Cold,
    Weak,
    Internal,
    External,
    Custom(String),
}

/// A function in the IR module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub blocks: IndexMap<String, BasicBlock>,
    pub attributes: Vec<FunctionAttribute>,
    pub is_declaration: bool,
    pub is_vararg: bool,
    pub linkage: String,
    pub source_file: Option<String>,
    pub line_number: Option<u32>,
}

impl Function {
    pub fn new(name: impl Into<String>, return_type: Type) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            return_type,
            blocks: IndexMap::new(),
            attributes: Vec::new(),
            is_declaration: false,
            is_vararg: false,
            linkage: "external".into(),
            source_file: None,
            line_number: None,
        }
    }

    /// Add a parameter.
    pub fn add_param(&mut self, name: impl Into<String>, ty: Type) {
        self.params.push((name.into(), ty));
    }

    /// Add a basic block and return a mutable reference to it.
    pub fn add_block(&mut self, name: impl Into<String>) -> &mut BasicBlock {
        let name = name.into();
        self.blocks.entry(name.clone()).or_insert_with(|| BasicBlock::new(name.clone()));
        self.blocks.get_mut(&name).unwrap()
    }

    /// Get the entry block.
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.values().next()
    }

    /// Get a block by name.
    pub fn block(&self, name: &str) -> Option<&BasicBlock> {
        self.blocks.get(name)
    }

    /// Get a mutable block by name.
    pub fn block_mut(&mut self, name: &str) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(name)
    }

    /// Total number of instructions across all blocks.
    pub fn instruction_count(&self) -> usize {
        self.blocks.values().map(|b| b.instructions.len()).sum()
    }

    /// Iterate all instructions with their block name and index.
    pub fn instructions(&self) -> impl Iterator<Item = (&str, usize, &Instruction)> {
        self.blocks.iter().flat_map(|(bname, block)| {
            block.instructions.iter().enumerate().map(move |(i, instr)| {
                (bname.as_str(), i, instr)
            })
        })
    }

    /// All call instructions in this function.
    pub fn call_instructions(&self) -> Vec<(&str, usize, &Instruction)> {
        self.instructions().filter(|(_, _, i)| i.is_call()).collect()
    }

    /// Get the type signature of this function.
    pub fn function_type(&self) -> Type {
        let param_tys: Vec<Type> = self.params.iter().map(|(_, t)| t.clone()).collect();
        if self.is_vararg {
            Type::vararg_func(self.return_type.clone(), param_tys)
        } else {
            Type::func(self.return_type.clone(), param_tys)
        }
    }

    /// Compute predecessor information from successor edges.
    pub fn compute_predecessors(&mut self) {
        let mut pred_map: HashMap<String, Vec<String>> = HashMap::new();
        for (bname, block) in &self.blocks {
            for succ in &block.successors {
                pred_map.entry(succ.clone()).or_default().push(bname.clone());
            }
        }
        for (bname, block) in &mut self.blocks {
            block.predecessors = pred_map.remove(bname).unwrap_or_default();
        }
    }

    /// Whether this function looks like a protocol negotiation function.
    pub fn is_negotiation_relevant(&self) -> bool {
        let name_lower = self.name.to_lowercase();
        let patterns = [
            "cipher", "version", "negotiate", "handshake", "kex",
            "extension", "select", "choose", "ssl_method", "tls_process",
            "tls_construct", "ssl_do", "ssl_set", "ssl_ctx",
        ];
        patterns.iter().any(|p| name_lower.contains(p))
    }
}

// ---------------------------------------------------------------------------
// Global Variable
// ---------------------------------------------------------------------------

/// A global variable in the module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalVariable {
    pub name: String,
    pub ty: Type,
    pub initializer: Option<Value>,
    pub is_constant: bool,
    pub linkage: String,
    pub alignment: Option<u32>,
    pub section: Option<String>,
}

impl GlobalVariable {
    pub fn new(name: impl Into<String>, ty: Type) -> Self {
        Self {
            name: name.into(),
            ty,
            initializer: None,
            is_constant: false,
            linkage: "external".into(),
            alignment: None,
            section: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Type Definitions
// ---------------------------------------------------------------------------

/// A named struct type definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDef {
    pub name: String,
    pub ty: Type,
    pub is_opaque: bool,
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// IR metadata node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataNode {
    String(String),
    Value(Value),
    Tuple(Vec<MetadataNode>),
    Named { name: String, operands: Vec<MetadataNode> },
    DILocation { line: u32, column: u32, scope: String },
    DISubprogram { name: String, file: String, line: u32 },
    Null,
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// A complete LLVM IR module representing a protocol library (e.g., libssl).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub name: String,
    pub source_filename: Option<String>,
    pub target_triple: Option<String>,
    pub data_layout: Option<String>,
    pub functions: IndexMap<String, Function>,
    pub globals: IndexMap<String, GlobalVariable>,
    pub type_defs: IndexMap<String, TypeDef>,
    pub metadata: HashMap<String, MetadataNode>,
    pub module_flags: Vec<(String, String)>,
}

impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source_filename: None,
            target_triple: None,
            data_layout: None,
            functions: IndexMap::new(),
            globals: IndexMap::new(),
            type_defs: IndexMap::new(),
            metadata: HashMap::new(),
            module_flags: Vec::new(),
        }
    }

    /// Load from a serialised JSON representation.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Add a function to the module.
    pub fn add_function(&mut self, func: Function) {
        self.functions.insert(func.name.clone(), func);
    }

    /// Add a global variable.
    pub fn add_global(&mut self, global: GlobalVariable) {
        self.globals.insert(global.name.clone(), global);
    }

    /// Add a type definition.
    pub fn add_type_def(&mut self, td: TypeDef) {
        self.type_defs.insert(td.name.clone(), td);
    }

    /// Get a function by name.
    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    /// Get a mutable function by name.
    pub fn function_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.get_mut(name)
    }

    /// Get a global by name.
    pub fn global(&self, name: &str) -> Option<&GlobalVariable> {
        self.globals.get(name)
    }

    /// Total number of instructions in the module.
    pub fn total_instructions(&self) -> usize {
        self.functions.values().map(|f| f.instruction_count()).sum()
    }

    /// Find all functions matching a pattern.
    pub fn find_functions(&self, pattern: &str) -> Vec<&Function> {
        let regex = regex::Regex::new(pattern).ok();
        self.functions.values().filter(|f| {
            if let Some(ref re) = regex {
                re.is_match(&f.name)
            } else {
                f.name.contains(pattern)
            }
        }).collect()
    }

    /// Return all functions that are negotiation-relevant.
    pub fn negotiation_functions(&self) -> Vec<&Function> {
        self.functions.values().filter(|f| f.is_negotiation_relevant()).collect()
    }

    /// Compute predecessors for all functions.
    pub fn compute_all_predecessors(&mut self) {
        for func in self.functions.values_mut() {
            func.compute_predecessors();
        }
    }

    /// Get a hash of the module contents for cache invalidation.
    pub fn content_hash(&self) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        for (name, func) in &self.functions {
            hasher.update(name.as_bytes());
            hasher.update(&func.instruction_count().to_le_bytes());
        }
        for (name, _) in &self.globals {
            hasher.update(name.as_bytes());
        }
        hex::encode(hasher.finalize())
    }

    /// Build a simple module for testing.
    pub fn test_module() -> Self {
        let mut module = Module::new("test_module");
        module.target_triple = Some("x86_64-unknown-linux-gnu".into());

        // Create SSL_do_handshake function
        let mut handshake = Function::new("SSL_do_handshake", Type::i32());
        handshake.add_param("ssl", Type::ptr(Type::NamedStruct("SSL".into())));
        {
            let entry = handshake.add_block("entry");
            entry.push(Instruction::Alloca {
                dest: "retval".into(),
                ty: Type::i32(),
                num_elements: None,
                align: Some(4),
            });
            entry.push(Instruction::Load {
                dest: "method".into(),
                ty: Type::ptr(Type::NamedStruct("SSL_METHOD".into())),
                ptr: Value::reg("ssl", Type::ptr(Type::NamedStruct("SSL".into()))),
                volatile: false,
                align: Some(8),
            });
            entry.push(Instruction::Call {
                dest: Some("result".into()),
                func: Value::reg("method", Type::ptr(Type::func(Type::i32(), vec![]))),
                args: vec![Value::reg("ssl", Type::ptr(Type::NamedStruct("SSL".into())))],
                ret_ty: Type::i32(),
                is_tail: false,
                calling_conv: None,
                attrs: vec![],
            });
            entry.push(Instruction::Ret { value: Some(Value::reg("result", Type::i32())) });
        }
        module.add_function(handshake);

        // Create ssl3_choose_cipher
        let mut choose = Function::new("ssl3_choose_cipher", Type::ptr(Type::NamedStruct("SSL_CIPHER".into())));
        choose.add_param("s", Type::ptr(Type::NamedStruct("SSL".into())));
        choose.add_param("clnt", Type::ptr(Type::NamedStruct("STACK_OF_SSL_CIPHER".into())));
        {
            let entry = choose.add_block("entry");
            entry.push(Instruction::Alloca {
                dest: "selected".into(),
                ty: Type::ptr(Type::NamedStruct("SSL_CIPHER".into())),
                num_elements: None,
                align: Some(8),
            });
            entry.push(Instruction::CondBr {
                cond: Value::int(1, 1),
                true_dest: "select_loop".into(),
                false_dest: "fallback".into(),
            });

            let loop_bb = choose.add_block("select_loop");
            loop_bb.push(Instruction::Call {
                dest: Some("cipher".into()),
                func: Value::func_ref("sk_SSL_CIPHER_value", Type::func(
                    Type::ptr(Type::NamedStruct("SSL_CIPHER".into())),
                    vec![Type::ptr(Type::NamedStruct("STACK_OF_SSL_CIPHER".into())), Type::i32()],
                )),
                args: vec![
                    Value::reg("clnt", Type::ptr(Type::NamedStruct("STACK_OF_SSL_CIPHER".into()))),
                    Value::int(0, 32),
                ],
                ret_ty: Type::ptr(Type::NamedStruct("SSL_CIPHER".into())),
                is_tail: false,
                calling_conv: None,
                attrs: vec![],
            });
            loop_bb.push(Instruction::Ret {
                value: Some(Value::reg("cipher", Type::ptr(Type::NamedStruct("SSL_CIPHER".into())))),
            });

            let fallback = choose.add_block("fallback");
            fallback.push(Instruction::Ret {
                value: Some(Value::null(Type::NamedStruct("SSL_CIPHER".into()))),
            });
        }
        module.add_function(choose);

        module.compute_all_predecessors();
        module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_display() {
        assert_eq!(Type::i32().to_string(), "i32");
        assert_eq!(Type::ptr(Type::i8()).to_string(), "i8*");
        assert_eq!(Type::array(Type::i64(), 4).to_string(), "[4 x i64]");
    }

    #[test]
    fn test_type_size() {
        assert_eq!(Type::i32().size_bits(), Some(32));
        assert_eq!(Type::Pointer(Box::new(Type::Void)).size_bits(), Some(64));
        assert_eq!(Type::array(Type::i8(), 16).size_bits(), Some(128));
    }

    #[test]
    fn test_value_construction() {
        let r = Value::reg("x", Type::i32());
        assert_eq!(r.name(), Some("x"));
        assert!(!r.is_constant());

        let c = Value::int(42, 32);
        assert!(c.is_constant());
    }

    #[test]
    fn test_instruction_dest_and_operands() {
        let load = Instruction::Load {
            dest: "val".into(),
            ty: Type::i32(),
            ptr: Value::reg("ptr", Type::ptr(Type::i32())),
            volatile: false,
            align: Some(4),
        };
        assert_eq!(load.dest(), Some("val"));
        assert_eq!(load.operands().len(), 1);
        assert!(!load.is_terminator());
        assert!(load.accesses_memory());
    }

    #[test]
    fn test_basic_block() {
        let mut bb = BasicBlock::new("entry");
        bb.push(Instruction::Alloca {
            dest: "x".into(),
            ty: Type::i32(),
            num_elements: None,
            align: Some(4),
        });
        bb.push(Instruction::Ret { value: None });
        assert_eq!(bb.len(), 2);
        assert!(bb.terminator().is_some());
        assert_eq!(bb.body().len(), 1);
        assert_eq!(bb.defs(), vec!["x"]);
    }

    #[test]
    fn test_function_creation() {
        let mut f = Function::new("test_func", Type::Void);
        f.add_param("arg0", Type::i32());
        let entry = f.add_block("entry");
        entry.push(Instruction::Ret { value: None });
        assert_eq!(f.instruction_count(), 1);
        assert!(f.entry_block().is_some());
    }

    #[test]
    fn test_module_test_module() {
        let module = Module::test_module();
        assert_eq!(module.functions.len(), 2);
        assert!(module.function("SSL_do_handshake").is_some());
        assert!(module.function("ssl3_choose_cipher").is_some());
        let neg = module.negotiation_functions();
        assert!(!neg.is_empty());
    }

    #[test]
    fn test_module_content_hash() {
        let m1 = Module::test_module();
        let m2 = Module::test_module();
        assert_eq!(m1.content_hash(), m2.content_hash());
    }

    #[test]
    fn test_instruction_successor_labels() {
        let br = Instruction::CondBr {
            cond: Value::int(1, 1),
            true_dest: "then".into(),
            false_dest: "else".into(),
        };
        let succs = br.successor_labels();
        assert_eq!(succs, vec!["then", "else"]);
    }

    #[test]
    fn test_call_instruction() {
        let call = Instruction::Call {
            dest: Some("ret".into()),
            func: Value::func_ref("SSL_read", Type::func(Type::i32(), vec![])),
            args: vec![Value::reg("ssl", Type::ptr(Type::Void))],
            ret_ty: Type::i32(),
            is_tail: false,
            calling_conv: None,
            attrs: vec![],
        };
        assert_eq!(call.called_function_name(), Some("SSL_read"));
        assert!(call.is_call());
    }

    #[test]
    fn test_function_negotiation_relevant() {
        let f = Function::new("ssl3_choose_cipher", Type::Void);
        assert!(f.is_negotiation_relevant());

        let f2 = Function::new("BIO_read", Type::Void);
        assert!(!f2.is_negotiation_relevant());
    }

    #[test]
    fn test_module_json_roundtrip() {
        let module = Module::new("test");
        let json = module.to_json().unwrap();
        let loaded = Module::from_json(&json).unwrap();
        assert_eq!(loaded.name, "test");
    }

    #[test]
    fn test_phi_operands() {
        let phi = Instruction::Phi {
            dest: "merged".into(),
            ty: Type::i32(),
            incoming: vec![
                (Value::int(1, 32), "bb1".into()),
                (Value::int(2, 32), "bb2".into()),
            ],
        };
        assert_eq!(phi.operands().len(), 2);
        assert_eq!(phi.dest(), Some("merged"));
    }

    #[test]
    fn test_switch_successors() {
        let sw = Instruction::Switch {
            value: Value::reg("v", Type::i32()),
            default_dest: "default".into(),
            cases: vec![
                (Value::int(0, 32), "case0".into()),
                (Value::int(1, 32), "case1".into()),
            ],
        };
        let succs = sw.successor_labels();
        assert_eq!(succs.len(), 3);
        assert_eq!(succs[0], "default");
    }

    #[test]
    fn test_find_functions() {
        let module = Module::test_module();
        let found = module.find_functions("ssl");
        assert!(!found.is_empty());
    }
}
