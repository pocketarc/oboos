//! Compile-time bridge between Rust types and store values.
//!
//! The [`StoreValue`] trait converts between concrete Rust types and the
//! dynamically-typed [`Value`]/[`FieldKind`] layer. This gives kernel code
//! type-safe store access without runtime string matching or manual
//! variant extraction.
//!
//! ## Scalar types
//!
//! Implementations are provided for all scalar types the store supports:
//! `bool`, `u8`, `u32`, `u64`, `i64`, and [`String`]. Each maps to its
//! corresponding [`Value`] and [`FieldKind`] variant.
//!
//! ## Collections
//!
//! Full `StoreValue` for `Vec<T>` is blocked by the `&'static FieldKind`
//! problem — `field_kind()` can't produce a static reference from a generic.
//! Helper functions [`vec_into_list`] and [`list_into_vec`] cover the common
//! case. A derive macro can generate the static references later.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::schema::FieldKind;
use crate::value::Value;

/// Bridge between a concrete Rust type and the store's dynamic type system.
///
/// Implement this to make a type usable with the typed `Store<S>` handle.
///
/// # Examples
///
/// ```
/// assert_eq!(u32::field_kind(), FieldKind::U32);
/// assert_eq!(u32::into_value(42), Value::U32(42));
/// assert_eq!(u32::from_value(&Value::U32(42)), Some(42));
/// ```
pub trait StoreValue: Sized {
    /// The [`FieldKind`] that corresponds to this type.
    fn field_kind() -> FieldKind;

    /// Convert this value into a [`Value`].
    fn into_value(self) -> Value;

    /// Try to extract this type from a [`Value`]. Returns `None` on
    /// variant mismatch.
    fn from_value(value: &Value) -> Option<Self>;
}

impl StoreValue for bool {
    fn field_kind() -> FieldKind { FieldKind::Bool }
    fn into_value(self) -> Value { Value::Bool(self) }
    fn from_value(value: &Value) -> Option<Self> {
        match value { Value::Bool(v) => Some(*v), _ => None }
    }
}

impl StoreValue for u8 {
    fn field_kind() -> FieldKind { FieldKind::U8 }
    fn into_value(self) -> Value { Value::U8(self) }
    fn from_value(value: &Value) -> Option<Self> {
        match value { Value::U8(v) => Some(*v), _ => None }
    }
}

impl StoreValue for u32 {
    fn field_kind() -> FieldKind { FieldKind::U32 }
    fn into_value(self) -> Value { Value::U32(self) }
    fn from_value(value: &Value) -> Option<Self> {
        match value { Value::U32(v) => Some(*v), _ => None }
    }
}

impl StoreValue for u64 {
    fn field_kind() -> FieldKind { FieldKind::U64 }
    fn into_value(self) -> Value { Value::U64(self) }
    fn from_value(value: &Value) -> Option<Self> {
        match value { Value::U64(v) => Some(*v), _ => None }
    }
}

impl StoreValue for i64 {
    fn field_kind() -> FieldKind { FieldKind::I64 }
    fn into_value(self) -> Value { Value::I64(self) }
    fn from_value(value: &Value) -> Option<Self> {
        match value { Value::I64(v) => Some(*v), _ => None }
    }
}

impl StoreValue for String {
    fn field_kind() -> FieldKind { FieldKind::Str }
    fn into_value(self) -> Value { Value::Str(self) }
    fn from_value(value: &Value) -> Option<Self> {
        match value { Value::Str(s) => Some(s.clone()), _ => None }
    }
}

/// Convert a `Vec<T>` into a `Value::List`.
///
/// This is a free function rather than a `StoreValue` impl because
/// `Vec<T>::field_kind()` would need to return `FieldKind::List(&'static ...)`
/// — a static reference that can't be produced from a generic. Use this
/// helper for the common case; a derive macro solves it for concrete types.
pub fn vec_into_list<T: StoreValue>(items: Vec<T>) -> Value {
    Value::List(items.into_iter().map(|item| item.into_value()).collect())
}

/// Try to extract a `Vec<T>` from a `Value::List`.
///
/// Returns `None` if the value isn't a List or if any element fails to
/// convert. See [`vec_into_list`] for the rationale on why this is a
/// free function.
pub fn list_into_vec<T: StoreValue>(value: &Value) -> Option<Vec<T>> {
    match value {
        Value::List(items) => {
            items.iter().map(|item| T::from_value(item)).collect()
        }
        _ => None,
    }
}
