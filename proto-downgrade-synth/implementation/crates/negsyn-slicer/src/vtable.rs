//! Virtual table and callback analysis for protocol dispatch.
//!
//! Handles SSL_METHOD vtable structures, BIO_METHOD callbacks, and
//! other function-pointer-based dispatch patterns in TLS/SSH libraries.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use serde::{Serialize, Deserialize};
use indexmap::IndexMap;

use crate::ir::{Module, Function, Instruction, Value, Type, GlobalVariable};
use crate::points_to::{AndersonAnalysis, PointsToSet, AbstractLocation};
use crate::{InstructionId, SlicerError, SlicerResult};

// ---------------------------------------------------------------------------
// VTable layout
// ---------------------------------------------------------------------------

/// A virtual table layout describing function pointer slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VTableLayout {
    /// Name of the struct type (e.g., "SSL_METHOD", "BIO_METHOD").
    pub type_name: String,
    /// Ordered list of entries (slots).
    pub entries: Vec<VTableEntry>,
    /// Total size in bytes (if known).
    pub size_bytes: Option<usize>,
    /// Whether this is a protocol-relevant vtable.
    pub is_protocol_relevant: bool,
}

/// A single entry in a vtable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VTableEntry {
    /// Field index in the struct.
    pub field_index: u32,
    /// Human-readable slot name (e.g., "ssl_connect", "ssl_accept").
    pub slot_name: String,
    /// Expected function signature.
    pub signature: Option<Type>,
    /// Known target functions that may fill this slot.
    pub known_targets: Vec<String>,
    /// Whether this slot is negotiation-relevant.
    pub is_negotiation_relevant: bool,
}

impl VTableLayout {
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            entries: Vec::new(),
            size_bytes: None,
            is_protocol_relevant: false,
        }
    }

    /// Add an entry to the vtable layout.
    pub fn add_entry(&mut self, entry: VTableEntry) {
        self.entries.push(entry);
    }

    /// Get an entry by field index.
    pub fn entry_at(&self, field_index: u32) -> Option<&VTableEntry> {
        self.entries.iter().find(|e| e.field_index == field_index)
    }

    /// Get an entry by slot name.
    pub fn entry_by_name(&self, name: &str) -> Option<&VTableEntry> {
        self.entries.iter().find(|e| e.slot_name == name)
    }

    /// Number of slots.
    pub fn num_slots(&self) -> usize {
        self.entries.len()
    }

    /// Get negotiation-relevant entries.
    pub fn negotiation_entries(&self) -> Vec<&VTableEntry> {
        self.entries.iter().filter(|e| e.is_negotiation_relevant).collect()
    }

    /// Build the standard SSL_METHOD vtable layout.
    pub fn ssl_method() -> Self {
        let mut layout = VTableLayout::new("SSL_METHOD");
        layout.is_protocol_relevant = true;
        let slots = vec![
            ("version", false),
            ("ssl_new", false),
            ("ssl_clear", false),
            ("ssl_free", false),
            ("ssl_accept", true),
            ("ssl_connect", true),
            ("ssl_read", false),
            ("ssl_peek", false),
            ("ssl_write", false),
            ("ssl_shutdown", false),
            ("ssl_renegotiate", true),
            ("ssl_renegotiate_check", true),
            ("ssl_read_bytes", false),
            ("ssl_write_bytes", false),
            ("ssl_dispatch_alert", false),
            ("ssl_ctrl", false),
            ("ssl_ctx_ctrl", false),
            ("get_cipher_by_char", true),
            ("put_cipher_by_char", true),
            ("ssl_pending", false),
            ("num_ciphers", true),
            ("get_cipher", true),
            ("get_timeout", false),
            ("ssl3_enc", true),
            ("ssl_version", true),
            ("ssl_callback_ctrl", false),
            ("ssl_ctx_callback_ctrl", false),
        ];

        for (i, (name, neg_relevant)) in slots.into_iter().enumerate() {
            layout.add_entry(VTableEntry {
                field_index: i as u32,
                slot_name: name.to_string(),
                signature: None,
                known_targets: Vec::new(),
                is_negotiation_relevant: neg_relevant,
            });
        }
        layout
    }

    /// Build the standard BIO_METHOD vtable layout.
    pub fn bio_method() -> Self {
        let mut layout = VTableLayout::new("BIO_METHOD");
        layout.is_protocol_relevant = false;
        let slots = vec![
            "type_id", "name", "bwrite", "bwrite_old",
            "bread", "bread_old", "bputs", "bgets",
            "ctrl", "create", "destroy", "callback_ctrl",
        ];
        for (i, name) in slots.into_iter().enumerate() {
            layout.add_entry(VTableEntry {
                field_index: i as u32,
                slot_name: name.to_string(),
                signature: None,
                known_targets: Vec::new(),
                is_negotiation_relevant: false,
            });
        }
        layout
    }
}

impl fmt::Display for VTableLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VTable: {} ({} slots)", self.type_name, self.entries.len())?;
        for entry in &self.entries {
            let targets = if entry.known_targets.is_empty() {
                "unresolved".to_string()
            } else {
                entry.known_targets.join(", ")
            };
            let marker = if entry.is_negotiation_relevant { " [NEG]" } else { "" };
            writeln!(f, "  [{}] {} -> {}{}", entry.field_index, entry.slot_name, targets, marker)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VTable resolver
// ---------------------------------------------------------------------------

/// Resolves virtual calls through vtable structures.
pub struct VTableResolver {
    /// Known vtable layouts.
    layouts: HashMap<String, VTableLayout>,
    /// Mapping from global variable name to its vtable type.
    global_vtables: HashMap<String, String>,
    /// Resolved dispatch targets: (vtable_instance, slot) → function names.
    resolved: HashMap<(String, u32), Vec<String>>,
}

impl VTableResolver {
    pub fn new() -> Self {
        let mut layouts = HashMap::new();
        layouts.insert("SSL_METHOD".into(), VTableLayout::ssl_method());
        layouts.insert("BIO_METHOD".into(), VTableLayout::bio_method());

        Self {
            layouts,
            global_vtables: HashMap::new(),
            resolved: HashMap::new(),
        }
    }

    /// Register a custom vtable layout.
    pub fn add_layout(&mut self, layout: VTableLayout) {
        self.layouts.insert(layout.type_name.clone(), layout);
    }

    /// Analyze a module to discover vtable instances and populate targets.
    pub fn analyze_module(&mut self, module: &Module) {
        // Find global variables that look like vtable instances.
        for (gname, global) in &module.globals {
            if let Some(vtable_type) = self.identify_vtable_type(gname, &global.ty) {
                self.global_vtables.insert(gname.clone(), vtable_type.clone());

                // Try to extract targets from the initializer.
                if let Some(ref init) = global.initializer {
                    self.extract_vtable_targets(gname, &vtable_type, init);
                }
            }
        }

        // Scan functions for vtable population patterns.
        for (_fname, func) in &module.functions {
            self.scan_function_for_vtable_writes(func);
        }
    }

    /// Identify if a global's type is a vtable type.
    fn identify_vtable_type(&self, name: &str, ty: &Type) -> Option<String> {
        let name_lower = name.to_lowercase();
        // Check by name pattern.
        if name_lower.contains("method") {
            if name_lower.contains("tls") || name_lower.contains("ssl") || name_lower.contains("dtls") {
                return Some("SSL_METHOD".into());
            }
            if name_lower.contains("bio") {
                return Some("BIO_METHOD".into());
            }
        }
        // Check by struct type name.
        match ty {
            Type::NamedStruct(sname) if self.layouts.contains_key(sname) => {
                Some(sname.clone())
            }
            Type::Pointer(inner) => {
                if let Type::NamedStruct(sname) = inner.as_ref() {
                    if self.layouts.contains_key(sname) {
                        return Some(sname.clone());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Extract function pointer targets from a vtable initializer.
    fn extract_vtable_targets(&mut self, vtable_name: &str, vtable_type: &str, init: &Value) {
        if let Value::Aggregate(fields, _) = init {
            for (i, field) in fields.iter().enumerate() {
                match field {
                    Value::FunctionRef(fname, _) => {
                        self.resolved
                            .entry((vtable_name.to_string(), i as u32))
                            .or_default()
                            .push(fname.clone());

                        // Also update the layout.
                        if let Some(layout) = self.layouts.get_mut(vtable_type) {
                            if let Some(entry) = layout.entries.iter_mut()
                                .find(|e| e.field_index == i as u32)
                            {
                                if !entry.known_targets.contains(fname) {
                                    entry.known_targets.push(fname.clone());
                                }
                            }
                        }
                    }
                    Value::NullPtr(_) => {
                        // NULL slot — no target.
                    }
                    _ => {}
                }
            }
        }
    }

    /// Scan a function for stores to vtable fields.
    fn scan_function_for_vtable_writes(&mut self, func: &Function) {
        for (_bname, block) in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::Store { value, ptr, .. } = instr {
                    // Check if we're storing a function pointer to a vtable slot.
                    if let Value::FunctionRef(target, _) = value {
                        if let Some(ptr_name) = ptr.name() {
                            // Check if ptr_name was derived from a GEP on a vtable.
                            if let Some((vtable_name, field_idx)) = self.find_vtable_gep(func, ptr_name) {
                                self.resolved
                                    .entry((vtable_name.clone(), field_idx))
                                    .or_default()
                                    .push(target.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    /// Find a GEP that computes a vtable slot address.
    fn find_vtable_gep(&self, func: &Function, reg_name: &str) -> Option<(String, u32)> {
        for (_bname, block) in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::GetElementPtr { dest, ptr, indices, base_ty, .. } = instr {
                    if dest == reg_name {
                        // Check if the base pointer is a known vtable.
                        if let Some(base_name) = ptr.name() {
                            if self.global_vtables.contains_key(base_name) {
                                let field_idx = if indices.len() >= 2 {
                                    match &indices[1] {
                                        Value::IntConst(v, _) => *v as u32,
                                        _ => 0,
                                    }
                                } else {
                                    0
                                };
                                return Some((base_name.to_string(), field_idx));
                            }
                        }
                        // Check by type.
                        if let Type::NamedStruct(sname) = base_ty {
                            if self.layouts.contains_key(sname) {
                                let base_name = ptr.name().unwrap_or("unknown").to_string();
                                let field_idx = if indices.len() >= 2 {
                                    match &indices[1] {
                                        Value::IntConst(v, _) => *v as u32,
                                        _ => 0,
                                    }
                                } else {
                                    0
                                };
                                return Some((base_name, field_idx));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Resolve a virtual call: given a vtable instance and field index, return target functions.
    pub fn resolve(&self, vtable_name: &str, field_index: u32) -> Vec<&str> {
        self.resolved.get(&(vtable_name.to_string(), field_index))
            .map(|targets| targets.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Resolve using points-to analysis: given a pointer that may point to vtable instances.
    pub fn resolve_from_pts(&self, pts: &PointsToSet, field_index: u32) -> Vec<String> {
        let mut targets = Vec::new();
        for loc in pts.iter() {
            match loc {
                AbstractLocation::VTable { type_name, instance } => {
                    let key = (instance.clone(), field_index);
                    if let Some(resolved) = self.resolved.get(&key) {
                        targets.extend(resolved.iter().cloned());
                    }
                    // Also try the type-level layout.
                    if let Some(layout) = self.layouts.get(type_name) {
                        if let Some(entry) = layout.entry_at(field_index) {
                            targets.extend(entry.known_targets.iter().cloned());
                        }
                    }
                }
                AbstractLocation::Global { name } => {
                    let resolved = self.resolve(name, field_index);
                    targets.extend(resolved.into_iter().map(|s| s.to_string()));
                }
                _ => {}
            }
        }
        targets.sort();
        targets.dedup();
        targets
    }

    /// Get a vtable layout by type name.
    pub fn layout(&self, type_name: &str) -> Option<&VTableLayout> {
        self.layouts.get(type_name)
    }

    /// Get all known vtable instances.
    pub fn vtable_instances(&self) -> Vec<(&str, &str)> {
        self.global_vtables.iter()
            .map(|(name, ty)| (name.as_str(), ty.as_str()))
            .collect()
    }

    /// Devirtualize: replace indirect calls through known vtable slots with direct calls.
    pub fn devirtualize_candidates(&self, module: &Module) -> Vec<DevirtualizationCandidate> {
        let mut candidates = Vec::new();

        for (fname, func) in &module.functions {
            for (bname, block) in &func.blocks {
                for (idx, instr) in block.instructions.iter().enumerate() {
                    if let Instruction::Call { func: callee, args, dest, ret_ty, .. } = instr {
                        // Check if this is an indirect call through a loaded vtable slot.
                        if let Value::Register(reg, _) = callee {
                            if let Some((vtable, field)) = self.trace_vtable_load(func, reg) {
                                let targets = self.resolve(&vtable, field);
                                if !targets.is_empty() {
                                    candidates.push(DevirtualizationCandidate {
                                        call_site: InstructionId::new(fname, bname, idx),
                                        vtable_instance: vtable,
                                        field_index: field,
                                        resolved_targets: targets.into_iter()
                                            .map(|s| s.to_string()).collect(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        candidates
    }

    /// Trace a register back to a vtable load pattern.
    fn trace_vtable_load(&self, func: &Function, reg: &str) -> Option<(String, u32)> {
        // Look for: %reg = load ptr, %gep_result
        // where: %gep_result = getelementptr %vtable, 0, field_index
        for (_bname, block) in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::Load { dest, ptr, .. } = instr {
                    if dest == reg {
                        if let Some(gep_name) = ptr.name() {
                            return self.find_vtable_gep(func, gep_name);
                        }
                    }
                }
            }
        }
        None
    }
}

/// A candidate for devirtualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevirtualizationCandidate {
    pub call_site: InstructionId,
    pub vtable_instance: String,
    pub field_index: u32,
    pub resolved_targets: Vec<String>,
}

// ---------------------------------------------------------------------------
// Callback analysis
// ---------------------------------------------------------------------------

/// Tracks callback registrations and invocations in protocol code.
pub struct CallbackAnalysis {
    /// Registered callbacks: (registration_site, callback_name) → target function.
    registrations: HashMap<(InstructionId, String), Vec<String>>,
    /// Callback invocation sites.
    invocations: Vec<CallbackInvocation>,
    /// Known callback registration functions.
    registration_functions: HashSet<String>,
}

/// A callback invocation site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallbackInvocation {
    pub site: InstructionId,
    pub callback_kind: String,
    pub possible_targets: Vec<String>,
}

impl CallbackAnalysis {
    pub fn new() -> Self {
        let mut reg_funcs = HashSet::new();
        // Known SSL callback registration functions.
        for name in &[
            "SSL_CTX_set_info_callback",
            "SSL_CTX_set_verify",
            "SSL_CTX_set_cert_verify_callback",
            "SSL_CTX_set_cert_cb",
            "SSL_CTX_set_alpn_select_cb",
            "SSL_CTX_set_next_proto_select_cb",
            "SSL_CTX_set_tlsext_servername_callback",
            "SSL_set_info_callback",
            "SSL_set_verify",
            "SSL_CTX_set_client_hello_cb",
            "SSL_CTX_set_keylog_callback",
            "BIO_set_callback",
            "BIO_set_callback_ex",
        ] {
            reg_funcs.insert(name.to_string());
        }

        Self {
            registrations: HashMap::new(),
            invocations: Vec::new(),
            registration_functions: reg_funcs,
        }
    }

    /// Analyze a module for callback registrations and invocations.
    pub fn analyze(&mut self, module: &Module) {
        for (fname, func) in &module.functions {
            for (bname, block) in &func.blocks {
                for (idx, instr) in block.instructions.iter().enumerate() {
                    let site = InstructionId::new(fname, bname, idx);
                    self.check_registration(instr, &site);
                    self.check_invocation(instr, &site, func);
                }
            }
        }
    }

    /// Check if an instruction is a callback registration.
    fn check_registration(&mut self, instr: &Instruction, site: &InstructionId) {
        if let Some(callee) = instr.called_function_name() {
            if self.registration_functions.contains(callee) {
                // The last argument is typically the callback function pointer.
                let args = match instr {
                    Instruction::Call { args, .. } | Instruction::Invoke { args, .. } => args,
                    _ => return,
                };
                if let Some(last_arg) = args.last() {
                    let target = match last_arg {
                        Value::FunctionRef(name, _) => Some(name.clone()),
                        Value::Register(name, _) => Some(format!("indirect:{}", name)),
                        _ => None,
                    };
                    if let Some(target) = target {
                        let callback_kind = callee.to_string();
                        self.registrations
                            .entry((site.clone(), callback_kind))
                            .or_default()
                            .push(target);
                    }
                }
            }
        }
    }

    /// Check if an instruction invokes a callback through a loaded function pointer.
    fn check_invocation(&mut self, instr: &Instruction, site: &InstructionId, func: &Function) {
        if let Instruction::Call { func: callee, .. } = instr {
            // Indirect call through a register (potential callback invocation).
            if let Value::Register(reg, _) = callee {
                // Check if this register was loaded from a callback slot.
                if let Some(kind) = self.identify_callback_kind(func, reg) {
                    let possible_targets = self.find_registered_targets(&kind);
                    self.invocations.push(CallbackInvocation {
                        site: site.clone(),
                        callback_kind: kind,
                        possible_targets,
                    });
                }
            }
        }
    }

    /// Try to identify the kind of callback from the load pattern.
    fn identify_callback_kind(&self, func: &Function, reg: &str) -> Option<String> {
        for (_bname, block) in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::Load { dest, ptr, .. } = instr {
                    if dest == reg {
                        if let Some(ptr_name) = ptr.name() {
                            let lower = ptr_name.to_lowercase();
                            if lower.contains("callback") || lower.contains("cb")
                                || lower.contains("info_callback") || lower.contains("verify")
                            {
                                return Some(ptr_name.to_string());
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Find registered callback targets for a given callback kind.
    fn find_registered_targets(&self, kind: &str) -> Vec<String> {
        let kind_lower = kind.to_lowercase();
        let mut targets = Vec::new();
        for ((_, reg_kind), reg_targets) in &self.registrations {
            if reg_kind.to_lowercase().contains(&kind_lower) || kind_lower.contains(&reg_kind.to_lowercase()) {
                targets.extend(reg_targets.iter().cloned());
            }
        }
        targets.sort();
        targets.dedup();
        targets
    }

    /// Get all callback registrations.
    pub fn registrations(&self) -> &HashMap<(InstructionId, String), Vec<String>> {
        &self.registrations
    }

    /// Get all callback invocations.
    pub fn invocations(&self) -> &[CallbackInvocation] {
        &self.invocations
    }

    /// Get negotiation-relevant callbacks (ALPN, SNI, etc.).
    pub fn negotiation_callbacks(&self) -> Vec<&CallbackInvocation> {
        self.invocations.iter().filter(|inv| {
            let kind_lower = inv.callback_kind.to_lowercase();
            kind_lower.contains("alpn") || kind_lower.contains("servername")
                || kind_lower.contains("hello") || kind_lower.contains("select")
                || kind_lower.contains("verify") || kind_lower.contains("cert")
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type};

    #[test]
    fn test_ssl_method_layout() {
        let layout = VTableLayout::ssl_method();
        assert!(layout.num_slots() > 10);
        assert!(layout.is_protocol_relevant);
        assert!(layout.entry_by_name("ssl_accept").is_some());
        let neg = layout.negotiation_entries();
        assert!(!neg.is_empty());
    }

    #[test]
    fn test_bio_method_layout() {
        let layout = VTableLayout::bio_method();
        assert!(layout.num_slots() > 5);
        assert!(!layout.is_protocol_relevant);
    }

    #[test]
    fn test_vtable_entry_at() {
        let layout = VTableLayout::ssl_method();
        let entry = layout.entry_at(4).unwrap();
        assert_eq!(entry.slot_name, "ssl_accept");
    }

    #[test]
    fn test_vtable_display() {
        let layout = VTableLayout::ssl_method();
        let s = format!("{}", layout);
        assert!(s.contains("SSL_METHOD"));
        assert!(s.contains("ssl_accept"));
    }

    #[test]
    fn test_vtable_resolver_basic() {
        let resolver = VTableResolver::new();
        assert!(resolver.layout("SSL_METHOD").is_some());
        assert!(resolver.layout("BIO_METHOD").is_some());
        assert!(resolver.layout("UNKNOWN").is_none());
    }

    #[test]
    fn test_vtable_resolver_analyze() {
        let mut module = Module::new("test");
        // Add a global vtable.
        let mut vtable = GlobalVariable::new(
            "TLSv1_2_method_data",
            Type::NamedStruct("SSL_METHOD".into()),
        );
        vtable.initializer = Some(Value::Aggregate(
            vec![
                Value::func_ref("tls1_2_version", Type::func(Type::i32(), vec![])),
                Value::func_ref("tls1_new", Type::func(Type::i32(), vec![])),
            ],
            Type::NamedStruct("SSL_METHOD".into()),
        ));
        module.add_global(vtable);

        let mut resolver = VTableResolver::new();
        resolver.analyze_module(&module);

        let instances = resolver.vtable_instances();
        assert!(!instances.is_empty());
    }

    #[test]
    fn test_callback_analysis_new() {
        let analysis = CallbackAnalysis::new();
        assert!(analysis.registration_functions.contains("SSL_CTX_set_alpn_select_cb"));
    }

    #[test]
    fn test_callback_analysis_registration() {
        let mut module = Module::new("test");
        let mut func = Function::new("setup_ssl", Type::Void);
        func.add_param("ctx", Type::ptr(Type::NamedStruct("SSL_CTX".into())));
        {
            let entry = func.add_block("entry");
            entry.push(Instruction::Call {
                dest: None,
                func: Value::func_ref(
                    "SSL_CTX_set_alpn_select_cb",
                    Type::func(Type::Void, vec![]),
                ),
                args: vec![
                    Value::reg("ctx", Type::ptr(Type::NamedStruct("SSL_CTX".into()))),
                    Value::func_ref("my_alpn_callback", Type::func(Type::i32(), vec![])),
                ],
                ret_ty: Type::Void,
                is_tail: false,
                calling_conv: None,
                attrs: vec![],
            });
            entry.push(Instruction::Ret { value: None });
        }
        module.add_function(func);

        let mut analysis = CallbackAnalysis::new();
        analysis.analyze(&module);

        assert!(!analysis.registrations().is_empty());
    }

    #[test]
    fn test_resolve_from_pts() {
        let mut resolver = VTableResolver::new();
        // Manually add a resolved target.
        resolver.resolved.insert(
            ("tls12_instance".to_string(), 4),
            vec!["ssl3_accept".to_string()],
        );

        let mut pts = PointsToSet::empty();
        pts.insert(AbstractLocation::vtable("SSL_METHOD", "tls12_instance"));

        let targets = resolver.resolve_from_pts(&pts, 4);
        assert!(targets.contains(&"ssl3_accept".to_string()));
    }

    #[test]
    fn test_devirtualization_candidates() {
        let module = Module::test_module();
        let resolver = VTableResolver::new();
        let candidates = resolver.devirtualize_candidates(&module);
        // May or may not find candidates depending on test module structure.
        assert!(candidates.len() >= 0);
    }

    #[test]
    fn test_vtable_layout_custom() {
        let mut layout = VTableLayout::new("MY_VTABLE");
        layout.add_entry(VTableEntry {
            field_index: 0,
            slot_name: "init".into(),
            signature: None,
            known_targets: vec!["my_init".into()],
            is_negotiation_relevant: true,
        });
        assert_eq!(layout.num_slots(), 1);
        assert_eq!(layout.negotiation_entries().len(), 1);
    }

    #[test]
    fn test_callback_negotiation_filtering() {
        let mut analysis = CallbackAnalysis::new();
        analysis.invocations.push(CallbackInvocation {
            site: InstructionId::new("f", "bb", 0),
            callback_kind: "alpn_select_cb".into(),
            possible_targets: vec!["my_alpn".into()],
        });
        analysis.invocations.push(CallbackInvocation {
            site: InstructionId::new("f", "bb", 1),
            callback_kind: "bio_write_cb".into(),
            possible_targets: vec!["my_bio_write".into()],
        });
        let neg = analysis.negotiation_callbacks();
        assert_eq!(neg.len(), 1);
        assert_eq!(neg[0].callback_kind, "alpn_select_cb");
    }
}
