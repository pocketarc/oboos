//! Dynamically-typed values that store fields hold.
//!
//! A [`Value`] is the runtime representation of a single field in a store.
//! It's an untyped enum — the store registry validates at runtime that each
//! value matches the [`FieldKind`] declared in the schema. The typed
//! `StoreHandle<S>` layer (future) will eliminate these runtime checks for
//! callers that go through the schema-aware API.

use alloc::collections::VecDeque;
use alloc::string::String;
use crate::schema::FieldKind;

/// A dynamically-typed store value.
///
/// No floats (no FPU context save yet), no Null (optionality comes in a
/// later layer with process stores). `Queue` is a FIFO collection — the
/// first non-scalar variant.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    U8(u8),
    U32(u32),
    U64(u64),
    I64(i64),
    Str(String),
    /// A FIFO queue of values, all matching the same inner [`FieldKind`].
    Queue(VecDeque<Value>),
}

impl Value {
    /// Check whether this value's variant matches the expected [`FieldKind`].
    ///
    /// Used by the store registry to reject type-mismatched writes at
    /// runtime. A `Value::U32(42)` matches `FieldKind::U32` but not
    /// `FieldKind::Bool`, for example.
    pub fn matches(&self, kind: &FieldKind) -> bool {
        match (self, kind) {
            (Value::Bool(_), FieldKind::Bool)
            | (Value::U8(_), FieldKind::U8)
            | (Value::U32(_), FieldKind::U32)
            | (Value::U64(_), FieldKind::U64)
            | (Value::I64(_), FieldKind::I64)
            | (Value::Str(_), FieldKind::Str) => true,
            // Empty queue matches any Queue(inner) — vacuous truth.
            (Value::Queue(items), FieldKind::Queue(inner_kind)) => {
                items.iter().all(|item| item.matches(inner_kind))
            }
            _ => false,
        }
    }
}
