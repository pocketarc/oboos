//! OBOOS API — shared types between the kernel and future userspace.
//!
//! This crate defines the vocabulary types that cross the kernel/userspace
//! boundary: store schemas, field values, and process specifications. It's
//! `#![no_std]` so it works on both sides of that boundary.

#![no_std]

extern crate alloc;

pub mod error;
pub mod process;
pub mod schema;
pub mod value;

pub use error::{
    is_error, ERR_INVALID_ARG, ERR_NOT_FOUND, ERR_THRESHOLD, ERR_TYPE_MISMATCH,
    ERR_UNKNOWN_FIELD, ERR_WOULD_BLOCK,
};
pub use process::ProcessSpec;
pub use schema::{FieldDef, FieldKind, StoreSchema};
pub use value::Value;
