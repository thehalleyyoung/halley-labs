//! Strongly-typed identifiers for various entities.
use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! define_id {
    ($name:ident, $prefix:expr) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(u64);

        impl $name {
            pub fn new(val: u64) -> Self {
                Self(val)
            }

            pub fn as_u64(self) -> u64 {
                self.0
            }

            pub fn next(self) -> Self {
                Self(self.0 + 1)
            }

            pub fn zero() -> Self {
                Self(0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", $prefix, self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }

        impl From<u64> for $name {
            fn from(val: u64) -> Self {
                Self(val)
            }
        }

        impl From<usize> for $name {
            fn from(val: usize) -> Self {
                Self(val as u64)
            }
        }
    };
}

define_id!(TransactionId, "T");
define_id!(OperationId, "Op");
define_id!(ItemId, "Item");
define_id!(ColumnId, "Col");
define_id!(TableId, "Tbl");
define_id!(LockId, "Lk");
define_id!(VersionId, "V");
define_id!(SnapshotId, "Snap");
define_id!(ConstraintId, "C");
define_id!(ScheduleStepId, "Step");
define_id!(WorkloadId, "W");

/// An allocator for sequential IDs.
#[derive(Debug, Clone)]
pub struct IdAllocator<T> {
    next: u64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: From<u64>> IdAllocator<T> {
    pub fn new() -> Self {
        Self {
            next: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_start(start: u64) -> Self {
        Self {
            next: start,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn allocate(&mut self) -> T {
        let id = self.next;
        self.next += 1;
        T::from(id)
    }

    pub fn peek_next(&self) -> u64 {
        self.next
    }

    pub fn count_allocated(&self) -> u64 {
        self.next
    }
}

impl<T: From<u64>> Default for IdAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_id_creation() {
        let id = TransactionId::new(42);
        assert_eq!(id.as_u64(), 42);
        assert_eq!(format!("{}", id), "T42");
        assert_eq!(format!("{:?}", id), "T(42)");
    }

    #[test]
    fn test_id_ordering() {
        let a = TransactionId::new(1);
        let b = TransactionId::new(2);
        assert!(a < b);
        assert_eq!(a.next(), b);
    }

    #[test]
    fn test_id_allocator() {
        let mut alloc = IdAllocator::<TransactionId>::new();
        let a = alloc.allocate();
        let b = alloc.allocate();
        assert_eq!(a.as_u64(), 0);
        assert_eq!(b.as_u64(), 1);
        assert_eq!(alloc.count_allocated(), 2);
    }

    #[test]
    fn test_id_allocator_with_start() {
        let mut alloc = IdAllocator::<OperationId>::with_start(100);
        let a = alloc.allocate();
        assert_eq!(a.as_u64(), 100);
    }

    #[test]
    fn test_id_equality() {
        let a = ItemId::new(5);
        let b = ItemId::new(5);
        let c = ItemId::new(6);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_id_from_usize() {
        let id = TableId::from(10usize);
        assert_eq!(id.as_u64(), 10);
    }
}
