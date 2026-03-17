//! ELF binary parsing via the `goblin` crate.
//!
//! Provides a richer alternative to [`ElfAdapter`](super::ElfAdapter) that
//! leverages `goblin`'s full ELF parser for symbol resolution, section
//! enumeration, dynamic linking information, and architecture detection.
//!
//! # Example
//!
//! ```rust,no_run
//! use leak_transform::goblin_adapter::GoblinElfAdapter;
//!
//! let adapter = GoblinElfAdapter::from_path("libcrypto.so").unwrap();
//! for func in adapter.functions() {
//!     println!("{}: 0x{:x}", func.name, func.address);
//! }
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use goblin::elf::Elf;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use shared_types::{FunctionId, VirtualAddress};

/// Errors from the goblin-based ELF adapter.
#[derive(Debug, Error)]
pub enum GoblinError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ELF parse error: {0}")]
    Parse(String),
    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
    #[error("no .text section found")]
    NoTextSection,
}

/// A discovered function from ELF symbol table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElfFunction {
    /// Symbol name.
    pub name: String,
    /// Virtual address of the function entry point.
    pub address: u64,
    /// Size in bytes (0 if unknown).
    pub size: u64,
    /// Whether this is a global (exported) symbol.
    pub is_global: bool,
    /// Section index the symbol belongs to.
    pub section_index: Option<usize>,
}

/// A section from the ELF binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElfSection {
    /// Section name (e.g., ".text", ".rodata").
    pub name: String,
    /// Virtual address.
    pub address: u64,
    /// Size in bytes.
    pub size: u64,
    /// Whether the section is executable.
    pub is_executable: bool,
    /// Whether the section is writable.
    pub is_writable: bool,
    /// File offset.
    pub offset: u64,
}

/// ELF adapter powered by the `goblin` crate for comprehensive binary parsing.
pub struct GoblinElfAdapter {
    /// Raw binary data.
    data: Vec<u8>,
    /// Discovered functions.
    functions: Vec<ElfFunction>,
    /// ELF sections.
    sections: Vec<ElfSection>,
    /// Entry point address.
    entry_point: u64,
    /// Architecture string (e.g., "x86_64").
    architecture: String,
    /// Whether the binary is position-independent (PIE/shared library).
    is_pie: bool,
}

impl GoblinElfAdapter {
    /// Parse an ELF binary from a file path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GoblinError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(data)
    }

    /// Parse an ELF binary from raw bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, GoblinError> {
        // Parse in a block so the borrow of `data` by `elf` is dropped
        // before we move `data` into the struct.
        let (functions, sections, entry_point, architecture, is_pie) = {
            let elf = Elf::parse(&data).map_err(|e| GoblinError::Parse(e.to_string()))?;

            let architecture = match elf.header.e_machine {
                goblin::elf::header::EM_X86_64 => "x86_64".to_string(),
                goblin::elf::header::EM_AARCH64 => "aarch64".to_string(),
                goblin::elf::header::EM_386 => "x86".to_string(),
                other => return Err(GoblinError::UnsupportedArch(format!("e_machine={other}"))),
            };

            let is_pie = elf.header.e_type == goblin::elf::header::ET_DYN;

            // Extract sections
            let sections: Vec<ElfSection> = elf
                .section_headers
                .iter()
                .filter_map(|sh| {
                    let name = elf.shdr_strtab.get_at(sh.sh_name)?.to_string();
                    Some(ElfSection {
                        name,
                        address: sh.sh_addr,
                        size: sh.sh_size,
                        is_executable: sh.is_executable(),
                        is_writable: sh.is_writable(),
                        offset: sh.sh_offset,
                    })
                })
                .collect();

            // Extract function symbols
            let mut functions = Vec::new();
            for sym in elf.syms.iter() {
                if sym.st_type() == goblin::elf::sym::STT_FUNC && sym.st_size > 0 {
                    if let Some(name) = elf.strtab.get_at(sym.st_name) {
                        functions.push(ElfFunction {
                            name: name.to_string(),
                            address: sym.st_value,
                            size: sym.st_size,
                            is_global: sym.st_bind() == goblin::elf::sym::STB_GLOBAL,
                            section_index: if sym.st_shndx != goblin::elf::section_header::SHN_UNDEF as usize {
                                Some(sym.st_shndx)
                            } else {
                                None
                            },
                        });
                    }
                }
            }

            // Also check dynamic symbols
            for sym in elf.dynsyms.iter() {
                if sym.st_type() == goblin::elf::sym::STT_FUNC && sym.st_size > 0 {
                    if let Some(name) = elf.dynstrtab.get_at(sym.st_name) {
                        if !functions.iter().any(|f| f.name == name) {
                            functions.push(ElfFunction {
                                name: name.to_string(),
                                address: sym.st_value,
                                size: sym.st_size,
                                is_global: true,
                                section_index: None,
                            });
                        }
                    }
                }
            }

            functions.sort_by_key(|f| f.address);
            let entry = elf.entry;

            (functions, sections, entry, architecture, is_pie)
        };

        Ok(Self {
            data,
            functions,
            sections,
            entry_point,
            architecture,
            is_pie,
        })
    }

    /// All discovered function symbols.
    pub fn functions(&self) -> &[ElfFunction] {
        &self.functions
    }

    /// All ELF sections.
    pub fn sections(&self) -> &[ElfSection] {
        &self.sections
    }

    /// The binary entry point address.
    pub fn entry_point(&self) -> u64 {
        self.entry_point
    }

    /// Architecture string.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Whether the binary is position-independent.
    pub fn is_pie(&self) -> bool {
        self.is_pie
    }

    /// Get the .text section.
    pub fn text_section(&self) -> Option<&ElfSection> {
        self.sections.iter().find(|s| s.name == ".text")
    }

    /// Get the raw bytes for a given virtual address range.
    pub fn bytes_at(&self, vaddr: u64, len: usize) -> Option<&[u8]> {
        // Find the section containing this address
        for section in &self.sections {
            if vaddr >= section.address && vaddr + len as u64 <= section.address + section.size {
                let file_offset = section.offset + (vaddr - section.address);
                let start = file_offset as usize;
                let end = start + len;
                if end <= self.data.len() {
                    return Some(&self.data[start..end]);
                }
            }
        }
        None
    }

    /// Find a function by name.
    pub fn find_function(&self, name: &str) -> Option<&ElfFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Find functions matching a prefix (e.g., "AES_" or "crypto_").
    pub fn find_functions_by_prefix(&self, prefix: &str) -> Vec<&ElfFunction> {
        self.functions.iter().filter(|f| f.name.starts_with(prefix)).collect()
    }

    /// Convert to a `shared_types::Program` for analysis.
    pub fn to_program(&self) -> shared_types::Program {
        let mut program = shared_types::Program::new(&self.architecture);
        for (i, func) in self.functions.iter().enumerate() {
            let fid = FunctionId::new(i as u32);
            let mut f = shared_types::Function::new(
                fid,
                &func.name,
                VirtualAddress(func.address),
            );
            f.size = func.size;
            program.add_function(f);
        }
        program
    }

    /// Total size of executable code.
    pub fn code_size(&self) -> u64 {
        self.sections.iter().filter(|s| s.is_executable).map(|s| s.size).sum()
    }

    /// Number of function symbols.
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goblin_error_display() {
        let err = GoblinError::NoTextSection;
        assert_eq!(err.to_string(), "no .text section found");
    }

    #[test]
    fn test_elf_function_struct() {
        let f = ElfFunction {
            name: "aes_encrypt".to_string(),
            address: 0x1000,
            size: 256,
            is_global: true,
            section_index: Some(1),
        };
        assert_eq!(f.name, "aes_encrypt");
        assert!(f.is_global);
    }
}
