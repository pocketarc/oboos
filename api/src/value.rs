//! Dynamically-typed values that store fields hold.
//!
//! A [`Value`] is the runtime representation of a single field in a store.
//! It's an untyped enum — the store registry validates at runtime that each
//! value matches the [`FieldKind`] declared in the schema. The typed
//! `StoreHandle<S>` layer (future) will eliminate these runtime checks for
//! callers that go through the schema-aware API.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;
use crate::schema::FieldKind;

/// A dynamically-typed store value.
///
/// No floats (no FPU context save yet), no Null (optionality comes in a
/// later layer with process stores). Collection variants:
///
/// - `Queue` — consumed on read (FIFO deque).
/// - `List` — retained on read (ordered vec).
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
    /// A retained ordered list of values, all matching the same inner [`FieldKind`].
    /// Unlike Queue, GET returns the full list without consuming elements.
    List(Vec<Value>),
    /// A structured value with named, typed fields — like a Rust struct
    /// represented as a map. Field names are strings, values are recursive.
    Record(BTreeMap<String, Value>),
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
            // Same vacuous-truth pattern for List.
            (Value::List(items), FieldKind::List(inner_kind)) => {
                items.iter().all(|item| item.matches(inner_kind))
            }
            // Record: strict validation — exactly the declared fields, each
            // matching its declared kind. No extra fields, no missing fields.
            (Value::Record(map), FieldKind::Record(field_defs)) => {
                if map.len() != field_defs.len() {
                    return false;
                }
                field_defs.iter().all(|def| {
                    map.get(def.name).is_some_and(|v| v.matches(&def.kind))
                })
            }
            _ => false,
        }
    }
}
