//! Locale-specific NLP transformation support.
//!
//! Provides morphological rules, agreement patterns, and transformation
//! constraints for different locales, enabling multi-language testing.

pub mod agreement_rules;
pub mod english;
pub mod locale_registry;
pub mod morphology;

pub use english::EnglishLocale;
pub use locale_registry::{Locale, LocaleId, LocaleRegistry};
