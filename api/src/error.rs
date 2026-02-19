//! Structured error codes for the syscall interface.
//!
//! Error codes live at the top of the `u64` address space — they can never
//! collide with valid byte counts returned by SYS_STORE_GET, since no
//! single GET response will ever approach `u64::MAX` bytes. This lets
//! userspace distinguish errors from success with a single threshold check:
//! `result > ERR_THRESHOLD` means error.
//!
//! The kernel maps internal [`StoreError`](crate::value) variants to these
//! codes at the syscall boundary. Userspace never sees Rust enums — just
//! `u64` values in RAX after `syscall`.

/// The store ID doesn't correspond to any live store.
pub const ERR_NOT_FOUND: u64 = u64::MAX;

/// The field name doesn't exist in the store's schema.
pub const ERR_UNKNOWN_FIELD: u64 = u64::MAX - 1;

/// The value's type doesn't match the field's declared kind.
pub const ERR_TYPE_MISMATCH: u64 = u64::MAX - 2;

/// Bad pointer, bad UTF-8, buffer too small, malformed packed buffer, etc.
pub const ERR_INVALID_ARG: u64 = u64::MAX - 3;

/// The operation would block (future use for non-blocking watch).
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
