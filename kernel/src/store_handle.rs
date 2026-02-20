//! Typed store handle — compile-time schema safety over the dynamic store API.
//!
//! [`Store<S>`] wraps a [`StoreId`] and uses the [`StoreSchema`] type parameter
//! to provide typed access to store fields. Instead of passing raw `Value`
//! enums and matching variants, callers use concrete Rust types:
//!
//! ```
//! let store = Store::<CounterSchema>::create(&[("count", Value::U32(0))])?;
//! store.set::<u32>("count", 42)?;
//! let count: u32 = store.get::<u32>("count")?;
//! ```
//!
//! Field names remain `&str` for now. A derive macro for field accessors
//! (`store.count().set(42)`) comes in a later phase.
//!
//! ## Kernel-only (for now)
//!
//! liboboos has no `alloc`, so the typed handle lives in the kernel. When
//! userspace gets a heap allocator, this can move to liboboos.

extern crate alloc;

use alloc::vec::Vec;
use core::future::Future;
use core::marker::PhantomData;

use oboos_api::{StoreSchema, StoreValue, Value};

use crate::store::{self, StoreError, StoreId};

/// A typed handle to a store instance, parameterized by its schema.
///
/// The schema type `S` is only used at compile time — no runtime cost.
/// The handle stores just the raw [`StoreId`] (a `u64`).
pub struct Store<S: StoreSchema> {
    id: StoreId,
    _schema: PhantomData<S>,
}

impl<S: StoreSchema> Store<S> {
    /// Create a new store instance from the schema with the given defaults.
    ///
    /// This is a typed wrapper around [`store::create::<S>()`]. The defaults
    /// are still untyped `Value` pairs because the schema's field set is
    /// defined at the trait level, not as individual type parameters.
    pub fn create(defaults: &[(&str, Value)]) -> Result<Self, StoreError> {
        let id = store::create::<S>(defaults)?;
        Ok(Store { id, _schema: PhantomData })
    }

    /// Wrap an existing [`StoreId`] as a typed handle.
    ///
    /// The caller asserts that `id` was created from schema `S`. No runtime
    /// check is performed — if the schema doesn't match, subsequent typed
    /// operations will see unexpected values (but won't violate memory safety).
    pub fn from_id(id: StoreId) -> Self {
        Store { id, _schema: PhantomData }
    }

    /// Access the raw [`StoreId`] for lower-level APIs.
    pub fn id(&self) -> StoreId {
        self.id
    }

    /// Read a field's value as a concrete Rust type.
    ///
    /// Returns `StoreError::TypeMismatch` if the stored value doesn't match
    /// the requested type `T`. This shouldn't happen if the schema and code
    /// agree, but guards against misuse.
    pub fn get<T: StoreValue>(&self, field: &str) -> Result<T, StoreError> {
        let value = store::get(self.id, field)?;
        T::from_value(&value).ok_or(StoreError::TypeMismatch)
    }

    /// Write a typed value to a field.
    pub fn set<T: StoreValue>(&self, field: &str, value: T) -> Result<(), StoreError> {
        store::set(self.id, &[(field, value.into_value())])
    }

    /// Push a typed value onto a Queue or List field.
    pub fn push<T: StoreValue>(&self, field: &str, value: T) -> Result<(), StoreError> {
        store::push(self.id, field, value.into_value())
    }

    /// Pop the front element from a Queue field as a concrete type.
    pub fn pop<T: StoreValue>(&self, field: &str) -> Result<Option<T>, StoreError> {
        match store::pop(self.id, field)? {
            Some(v) => Ok(Some(T::from_value(&v).ok_or(StoreError::TypeMismatch)?)),
            None => Ok(None),
        }
    }

    /// Get a List field's contents as a `Vec<T>`.
    ///
    /// Returns `StoreError::TypeMismatch` if the field isn't a List or if
    /// any element fails to convert.
    pub fn list<T: StoreValue>(&self, field: &str) -> Result<Vec<T>, StoreError> {
        let value = store::get(self.id, field)?;
        oboos_api::store_value::list_into_vec::<T>(&value).ok_or(StoreError::TypeMismatch)
    }

    /// Watch one or more fields for changes.
    ///
    /// Returns a one-shot future that resolves the next time any of the
    /// watched fields is written. Same semantics as [`store::watch()`].
    pub fn watch(
        &self,
        fields: &'static [&'static str],
    ) -> impl Future<Output = Result<(), StoreError>> + Send {
        store::watch(self.id, fields)
    }

    /// Destroy the underlying store instance.
    pub fn destroy(self) -> Result<(), StoreError> {
        store::destroy(self.id)
    }
}
