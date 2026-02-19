//! Structured error codes and types for the store syscall interface.
//!
//! Error codes live at the top of the `u64` address space — they can never
//! collide with valid byte counts returned by SYS_STORE_GET, since no
//! single GET response will ever approach `u64::MAX` bytes. This lets
//! userspace distinguish errors from success with a single threshold check:
//! `result > ERR_THRESHOLD` means error.
//!
//! [`StoreError`] is the Rust enum used by both the kernel and liboboos.
//! The raw `u64` constants below are the ABI — what actually travels in
//! registers across the SYSCALL boundary. [`StoreError::from_raw`] and
//! [`StoreError::to_raw`] convert between the two representations.

/// The store ID doesn't correspond to any live store.
pub const ERR_NOT_FOUND: u64 = u64::MAX;

/// The field name doesn't exist in the store's schema.
pub const ERR_UNKNOWN_FIELD: u64 = u64::MAX - 1;

/// The value's type doesn't match the field's declared kind.
pub const ERR_TYPE_MISMATCH: u64 = u64::MAX - 2;

/// Bad pointer, bad UTF-8, buffer too small, malformed packed buffer, etc.
pub const ERR_INVALID_ARG: u64 = u64::MAX - 3;

/// The operation would block (e.g. no free subscription slots).
pub const ERR_WOULD_BLOCK: u64 = u64::MAX - 4;

/// Threshold for error detection — any result >= this is an error.
/// Leaves headroom for 12 more error codes before colliding with
/// `ERR_WOULD_BLOCK`.
pub const ERR_THRESHOLD: u64 = u64::MAX - 15;

/// Check whether a syscall return value is an error code.
///
/// # Examples
///
/// ```
/// let result = store_get(id, "counter");
/// if is_error(result) {
///     // handle error
/// } else {
///     // result is the number of bytes written
/// }
/// ```
pub fn is_error(result: u64) -> bool {
    result > ERR_THRESHOLD
}

/// Errors returned by store operations.
///
/// Used on both sides of the syscall boundary: the kernel returns these
/// from store functions, and liboboos converts raw `u64` error codes
/// back into this enum via [`from_raw`](StoreError::from_raw).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// No store with this ID exists (destroyed or never created).
    NotFound,
    /// The field name doesn't exist in this store's schema.
    UnknownField,
    /// The value's type doesn't match the field's declared kind.
    TypeMismatch,
    /// Bad pointer, bad UTF-8, buffer too small, malformed packed buffer, etc.
    InvalidArg,
    /// The operation would block (e.g. no free subscription slots).
    WouldBlock,
}

impl StoreError {
    /// Convert a raw `u64` error code from the syscall ABI into a
    /// [`StoreError`]. Returns `None` if the value isn't a recognized
    /// error code.
    pub fn from_raw(raw: u64) -> Option<Self> {
        match raw {
            ERR_NOT_FOUND => Some(StoreError::NotFound),
            ERR_UNKNOWN_FIELD => Some(StoreError::UnknownField),
            ERR_TYPE_MISMATCH => Some(StoreError::TypeMismatch),
            ERR_INVALID_ARG => Some(StoreError::InvalidArg),
            ERR_WOULD_BLOCK => Some(StoreError::WouldBlock),
            _ => None,
        }
    }

    /// Convert this error to its raw `u64` representation for the
    /// syscall ABI.
    pub fn to_raw(self) -> u64 {
        match self {
            StoreError::NotFound => ERR_NOT_FOUND,
            StoreError::UnknownField => ERR_UNKNOWN_FIELD,
            StoreError::TypeMismatch => ERR_TYPE_MISMATCH,
            StoreError::InvalidArg => ERR_INVALID_ARG,
            StoreError::WouldBlock => ERR_WOULD_BLOCK,
        }
    }
}

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StoreError::NotFound => write!(f, "store not found"),
            StoreError::UnknownField => write!(f, "unknown field"),
            StoreError::TypeMismatch => write!(f, "type mismatch"),
            StoreError::InvalidArg => write!(f, "invalid argument"),
            StoreError::WouldBlock => write!(f, "would block"),
        }
    }
}
