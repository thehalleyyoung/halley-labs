//! Binary adapter layer for reading ELF binaries and raw byte blobs.
//!
//! Provides a trait-based abstraction over binary formats, exposing sections,
//! symbol tables, and function discovery so that the lifter can work
//! independently of the on-disk format.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use shared_types::{AddressRange, VirtualAddress};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by binary adapters.
#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("unsupported binary format: {0}")]
    UnsupportedFormat(String),

    #[error("section `{name}` not found")]
    SectionNotFound { name: String },

    #[error("invalid ELF header: {0}")]
    InvalidElf(String),

    #[error("I/O error reading binary: {0}")]
    Io(#[from] std::io::Error),

    #[error("no executable sections found")]
    NoExecutableSections,

    #[error("symbol table is empty or missing")]
    NoSymbolTable,

    #[error("adapter error: {0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// Section
// ---------------------------------------------------------------------------

/// A section extracted from a binary image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Section name (e.g. `.text`, `.rodata`).
    pub name: String,
    /// Virtual address range of this section when loaded.
    pub address_range: AddressRange,
    /// Raw bytes of the section.
    pub data: Vec<u8>,
    /// Whether the section is executable.
    pub executable: bool,
    /// Whether the section is writable.
    pub writable: bool,
    /// Whether the section is readable.
    pub readable: bool,
}

impl Section {
    pub fn new(name: &str, start: VirtualAddress, data: Vec<u8>) -> Self {
        let end = start.offset(data.len() as i64);
        Self {
            name: name.to_string(),
            address_range: AddressRange::new(start, end),
            data,
            executable: false,
            writable: false,
            readable: true,
        }
    }

    /// Mark this section as executable.
    pub fn with_executable(mut self) -> Self {
        self.executable = true;
        self
    }

    /// Mark this section as writable.
    pub fn with_writable(mut self) -> Self {
        self.writable = true;
        self
    }

    /// Size in bytes.
    pub fn size(&self) -> u64 {
        self.address_range.len()
    }

    /// Read a byte at the given virtual address.
    pub fn read_byte(&self, addr: VirtualAddress) -> Option<u8> {
        if !self.address_range.contains(addr) {
            return None;
        }
        let offset = addr.as_u64() - self.address_range.start.as_u64();
        self.data.get(offset as usize).copied()
    }

    /// Read a slice of bytes starting at `addr` with length `len`.
    pub fn read_bytes(&self, addr: VirtualAddress, len: usize) -> Option<&[u8]> {
        if !self.address_range.contains(addr) {
            return None;
        }
        let offset = (addr.as_u64() - self.address_range.start.as_u64()) as usize;
        self.data.get(offset..offset + len)
    }
}

// ---------------------------------------------------------------------------
// SymbolTable
// ---------------------------------------------------------------------------

/// A simplified symbol table extracted from a binary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolTable {
    /// Symbol name → virtual address.
    pub symbols: HashMap<String, VirtualAddress>,
    /// Virtual address → symbol name (reverse lookup).
    pub addresses: HashMap<VirtualAddress, String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a symbol.
    pub fn add_symbol(&mut self, name: &str, address: VirtualAddress) {
        self.symbols.insert(name.to_string(), address);
        self.addresses.insert(address, name.to_string());
    }

    /// Look up a symbol by name.
    pub fn lookup(&self, name: &str) -> Option<VirtualAddress> {
        self.symbols.get(name).copied()
    }

    /// Look up a symbol name by address.
    pub fn name_at(&self, addr: VirtualAddress) -> Option<&str> {
        self.addresses.get(&addr).map(|s| s.as_str())
    }

    /// Number of symbols.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether the symbol table is empty.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

// ---------------------------------------------------------------------------
// FunctionDiscovery
// ---------------------------------------------------------------------------

/// Discovered function entries from a binary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionDiscovery {
    /// Discovered function entry points (address → name).
    pub entries: Vec<FunctionEntry>,
}

/// A single discovered function entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionEntry {
    /// Entry-point virtual address.
    pub address: VirtualAddress,
    /// Symbol name, if known.
    pub name: Option<String>,
    /// Estimated size in bytes, if known.
    pub estimated_size: Option<u64>,
}

impl FunctionDiscovery {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a discovered function entry.
    pub fn add_entry(&mut self, address: VirtualAddress, name: Option<&str>, size: Option<u64>) {
        self.entries.push(FunctionEntry {
            address,
            name: name.map(String::from),
            estimated_size: size,
        });
    }

    /// Sort entries by address.
    pub fn sort_by_address(&mut self) {
        self.entries.sort_by_key(|e| e.address);
    }

    /// Number of discovered functions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BinaryAdapter trait
// ---------------------------------------------------------------------------

/// Abstraction over binary formats (ELF, raw bytes, etc.).
///
/// Implementors parse a binary image and expose sections, symbols, and
/// function entries for consumption by the lifter.
pub trait BinaryAdapter: std::fmt::Debug {
    /// Load and parse the binary from raw bytes.
    fn load(&mut self, data: &[u8]) -> Result<(), AdapterError>;

    /// Return all sections in the binary.
    fn sections(&self) -> &[Section];

    /// Return the section with the given name.
    fn section_by_name(&self, name: &str) -> Result<&Section, AdapterError> {
        self.sections()
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| AdapterError::SectionNotFound {
                name: name.to_string(),
            })
    }

    /// Return the executable sections.
    fn executable_sections(&self) -> Vec<&Section> {
        self.sections().iter().filter(|s| s.executable).collect()
    }

    /// Return the symbol table (if available).
    fn symbol_table(&self) -> &SymbolTable;

    /// Discover function entry points.
    fn discover_functions(&self) -> Result<FunctionDiscovery, AdapterError>;

    /// Read bytes at a given virtual address across all sections.
    fn read_bytes(&self, addr: VirtualAddress, len: usize) -> Option<Vec<u8>> {
        for section in self.sections() {
            if let Some(bytes) = section.read_bytes(addr, len) {
                return Some(bytes.to_vec());
            }
        }
        None
    }

    /// Return the entry-point virtual address of the binary (if known).
    fn entry_point(&self) -> Option<VirtualAddress>;
}

// ---------------------------------------------------------------------------
// ElfAdapter
// ---------------------------------------------------------------------------

/// Adapter for ELF binaries.
#[derive(Debug, Clone, Default)]
pub struct ElfAdapter {
    sections: Vec<Section>,
    symbol_table: SymbolTable,
    entry_point: Option<VirtualAddress>,
    loaded: bool,
}

impl ElfAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the adapter has successfully loaded a binary.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
}

impl BinaryAdapter for ElfAdapter {
    fn load(&mut self, data: &[u8]) -> Result<(), AdapterError> {
        // Minimal ELF magic check.
        if data.len() < 16 || &data[0..4] != b"\x7fELF" {
            return Err(AdapterError::InvalidElf(
                "missing or invalid ELF magic".to_string(),
            ));
        }

        // Stub: a real implementation would parse ELF headers, program
        // headers, and section headers using an ELF parsing library.
        log::info!("ElfAdapter: loaded {} bytes", data.len());
        self.loaded = true;
        Ok(())
    }

    fn sections(&self) -> &[Section] {
        &self.sections
    }

    fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    fn discover_functions(&self) -> Result<FunctionDiscovery, AdapterError> {
        if self.symbol_table.is_empty() {
            return Err(AdapterError::NoSymbolTable);
        }

        let mut discovery = FunctionDiscovery::new();
        for (name, &addr) in &self.symbol_table.symbols {
            discovery.add_entry(addr, Some(name), None);
        }
        discovery.sort_by_address();
        Ok(discovery)
    }

    fn entry_point(&self) -> Option<VirtualAddress> {
        self.entry_point
    }
}

// ---------------------------------------------------------------------------
// RawBytesAdapter
// ---------------------------------------------------------------------------

/// Adapter for flat raw byte blobs (no format metadata).
#[derive(Debug, Clone)]
pub struct RawBytesAdapter {
    sections: Vec<Section>,
    symbol_table: SymbolTable,
    base_address: VirtualAddress,
}

impl Default for RawBytesAdapter {
    fn default() -> Self {
        Self {
            sections: Vec::new(),
            symbol_table: SymbolTable::new(),
            base_address: VirtualAddress::ZERO,
        }
    }
}

impl RawBytesAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an adapter pre-configured with a base load address.
    pub fn with_base_address(base: VirtualAddress) -> Self {
        Self {
            base_address: base,
            ..Default::default()
        }
    }

    /// Manually add a symbol (useful when symbols come from an external source).
    pub fn add_symbol(&mut self, name: &str, addr: VirtualAddress) {
        self.symbol_table.add_symbol(name, addr);
    }
}

impl BinaryAdapter for RawBytesAdapter {
    fn load(&mut self, data: &[u8]) -> Result<(), AdapterError> {
        let section = Section::new(".text", self.base_address, data.to_vec())
            .with_executable();
        self.sections = vec![section];
        log::info!(
            "RawBytesAdapter: loaded {} bytes at base {:#x}",
            data.len(),
            self.base_address.as_u64()
        );
        Ok(())
    }

    fn sections(&self) -> &[Section] {
        &self.sections
    }

    fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    fn discover_functions(&self) -> Result<FunctionDiscovery, AdapterError> {
        // Without symbols, return the entire blob as a single function.
        if self.symbol_table.is_empty() {
            let mut discovery = FunctionDiscovery::new();
            if let Some(section) = self.sections.first() {
                discovery.add_entry(
                    section.address_range.start,
                    Some("_start"),
                    Some(section.size()),
                );
            }
            return Ok(discovery);
        }

        let mut discovery = FunctionDiscovery::new();
        for (name, &addr) in &self.symbol_table.symbols {
            discovery.add_entry(addr, Some(name), None);
        }
        discovery.sort_by_address();
        Ok(discovery)
    }

    fn entry_point(&self) -> Option<VirtualAddress> {
        Some(self.base_address)
    }
}
