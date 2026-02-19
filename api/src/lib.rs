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
pub mod value;

pub use error::{
    is_error, StoreError, ERR_INVALID_ARG, ERR_NOT_FOUND, ERR_THRESHOLD, ERR_TYPE_MISMATCH,
    ERR_UNKNOWN_FIELD, ERR_WOULD_BLOCK,
};
#[cfg(feature = "alloc")]
pub use process::ProcessSpec;
pub use schema::{FieldDef, FieldKind, StoreSchema};
#[cfg(feature = "alloc")]
pub use value::Value;
