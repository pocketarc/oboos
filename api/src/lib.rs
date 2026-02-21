//! OBOOS API — shared types between the kernel and future userspace.
//!
//! This crate defines the vocabulary types that cross the kernel/userspace
//! boundary: store schemas, field values, and process specifications. It's
//! `#![no_std]` so it works on both sides of that boundary.
//!
//! The `alloc` feature (enabled by default) brings in [`Value`],
//! [`ProcessSpec`], and their dependencies on `alloc::string::String` and
//! `alloc::collections::VecDeque`. Disable it for no-alloc environments
//! like userspace (which uses raw syscall buffers instead of heap types).

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
#[cfg(feature = "alloc")]
pub mod process;
pub mod schema;
#[cfg(feature = "alloc")]
pub mod store_value;
#[cfg(feature = "alloc")]
pub mod value;

pub use error::{
    is_error, StoreError, ERR_INVALID_ARG, ERR_NOT_FOUND, ERR_THRESHOLD, ERR_TYPE_MISMATCH,
    ERR_UNKNOWN_FIELD, ERR_WOULD_BLOCK,
};
#[cfg(feature = "alloc")]
pub use process::ProcessSpec;
pub use schema::{FieldDef, FieldKind, StoreSchema};
#[cfg(feature = "alloc")]
pub use store_value::StoreValue;
#[cfg(feature = "alloc")]
pub use value::Value;

// ————————————————————————————————————————————————————————————————————————————
// Syscall numbers — shared between kernel and userspace
// ————————————————————————————————————————————————————————————————————————————

/// Read fields from a store. Multi-field packed buffer protocol.
pub const SYS_STORE_GET: u64 = 0;
/// Write fields to a store. Multi-field packed buffer protocol.
pub const SYS_STORE_SET: u64 = 1;
/// Register a persistent watcher on a store field.
pub const SYS_SUBSCRIBE: u64 = 2;
/// Remove a persistent watcher by subscription ID.
pub const SYS_UNSUBSCRIBE: u64 = 3;
/// Sleep until at least one subscription fires.
pub const SYS_YIELD: u64 = 4;
/// Dispatch a typed mutation to a store's reducer.
pub const SYS_STORE_MUTATE: u64 = 5;

// ————————————————————————————————————————————————————————————————————————————
// Process mutation IDs — shared between kernel and userspace
// ————————————————————————————————————————————————————————————————————————————

/// Mutation ID for `MapHeap { pages: u64 }` — allocate and map heap pages.
/// Payload: 8 bytes (u64 page count, native endian).
/// Returns: virtual address of the mapped region.
pub const PROCESS_MUTATE_MAP_HEAP: u8 = 0;

/// Mutation ID for `Exit { code: u64 }` — terminate the current process.
/// Payload: 8 bytes (u64 exit code, native endian).
/// Returns: 0 (never actually returned to userspace).
pub const PROCESS_MUTATE_EXIT: u8 = 1;
