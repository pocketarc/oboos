//! Kernel store registry — reactive state trees for IPC.
//!
//! The store is OBOOS's future IPC primitive: a shared, schema-validated
//! state tree that processes read, write, and (eventually) subscribe to.
//! This Layer 0 implements the data structure groundwork — create, get,
//! set, destroy — with schema validation on every write. No subscriptions,
//! no capabilities, no persistence yet.
//!
//! ## Design
//!
//! Each store instance is created from a [`StoreSchema`] that declares
//! its fields and their types. The registry validates every `set()` call
//! against the schema: unknown fields and type mismatches are rejected
//! before any data changes. This catches bugs at the API boundary rather
//! than letting bad data propagate.
//!
//! Store IDs are monotonic `u64` values, never reused. This avoids the
//! ABA problem where a destroyed store's ID gets recycled and a stale
//! handle silently writes to a different store.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use core::fmt;

use oboos_api::{FieldDef, StoreSchema, Value};

/// Opaque store identifier — monotonic, never reused.
///
/// Typed `StoreHandle<S>` wraps this in a later layer to provide
/// compile-time schema guarantees. At Layer 0, callers pass raw IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StoreId(u64);

/// Errors returned by store operations.
#[derive(Debug)]
pub enum StoreError {
    /// No store with this ID exists (destroyed or never created).
    NotFound,
    /// The field name doesn't exist in this store's schema.
    UnknownField,
    /// The value's type doesn't match the field's declared kind.
    TypeMismatch,
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::NotFound => write!(f, "store not found"),
            StoreError::UnknownField => write!(f, "unknown field"),
            StoreError::TypeMismatch => write!(f, "type mismatch"),
        }
    }
}

/// A live store instance: schema metadata + current field values.
struct StoreInstance {
    fields: &'static [FieldDef],
    #[allow(dead_code)]
    schema_name: &'static str,
    data: BTreeMap<String, Value>,
}

/// The global store registry — all live store instances keyed by ID.
struct StoreRegistry {
    stores: BTreeMap<u64, StoreInstance>,
    next_id: u64,
}

/// Global registry, initialized once from `kmain` via [`init()`].
///
/// Uses `spin::Once<spin::Mutex<...>>` — same pattern as the scheduler
/// and executor. The `Once` ensures we panic on use-before-init rather
/// than silently operating on empty state.
static REGISTRY: spin::Once<spin::Mutex<StoreRegistry>> = spin::Once::new();

fn registry() -> &'static spin::Mutex<StoreRegistry> {
    REGISTRY.get().expect("store::init() not called")
}

/// Initialize the store registry. Must be called after [`crate::heap::init()`].
///
/// The store needs the heap (for BTreeMap) but has no dependency on the
/// scheduler or executor — call it early in the boot sequence.
pub fn init() {
    REGISTRY.call_once(|| {
        spin::Mutex::new(StoreRegistry {
            stores: BTreeMap::new(),
            next_id: 0,
        })
    });
    crate::println!("[ok] Store registry initialized");
}

/// Create a new store instance from a schema with the given default values.
///
/// Every field declared in the schema must have a corresponding default —
/// no uninitialized fields, ever. Panics if a default is missing or its
/// type doesn't match the schema (these are programming errors, not
/// runtime conditions).
///
/// # Examples
///
/// ```
/// let id = store::create::<CounterSchema>(&[
///     ("count", Value::U32(0)),
///     ("label", Value::Str(String::from("hello"))),
///     ("active", Value::Bool(true)),
/// ])?;
/// ```
pub fn create<S: StoreSchema>(defaults: &[(&str, Value)]) -> Result<StoreId, StoreError> {
    let fields = S::fields();
    let schema_name = S::name();

    // Build the data map from defaults, validating each one.
    let mut data = BTreeMap::new();
    for &(name, ref value) in defaults {
        let field = fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("default for unknown field '{name}' in schema '{schema_name}'"));
        assert!(
            value.matches(&field.kind),
            "default for field '{name}' has wrong type in schema '{schema_name}'"
        );
        data.insert(String::from(name), value.clone());
    }

    // Every schema field must have a default.
    for field in fields {
        assert!(
            data.contains_key(field.name),
            "missing default for field '{}' in schema '{}'",
            field.name,
            schema_name
        );
    }

    let mut reg = registry().lock();
    let id = reg.next_id;
    reg.next_id += 1;
    reg.stores.insert(
        id,
        StoreInstance {
            fields,
            schema_name,
            data,
        },
    );
    Ok(StoreId(id))
}

/// Read a field's current value (cloned out).
pub fn get(store: StoreId, field: &str) -> Result<Value, StoreError> {
    let reg = registry().lock();
    let instance = reg.stores.get(&store.0).ok_or(StoreError::NotFound)?;

    // Check the field exists in the schema, not just in the data map.
    // This catches typos even if the data map somehow had extra keys.
    if !instance.fields.iter().any(|f| f.name == field) {
        return Err(StoreError::UnknownField);
    }

    // The field is guaranteed to be in the data map because create()
    // requires a default for every schema field.
    Ok(instance.data[field].clone())
}

/// Write a new value to a field, with schema validation.
///
/// Rejects unknown fields ([`StoreError::UnknownField`]) and type
/// mismatches ([`StoreError::TypeMismatch`]) before mutating anything.
pub fn set(store: StoreId, field: &str, value: Value) -> Result<(), StoreError> {
    let mut reg = registry().lock();
    let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

    let field_def = instance
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or(StoreError::UnknownField)?;

    if !value.matches(&field_def.kind) {
        return Err(StoreError::TypeMismatch);
    }

    instance.data.insert(String::from(field), value);
    Ok(())
}

/// Destroy a store instance, freeing its data.
///
/// After this call, any `get`/`set`/`destroy` with this ID returns
/// [`StoreError::NotFound`]. The ID is never reused.
pub fn destroy(store: StoreId) -> Result<(), StoreError> {
    let mut reg = registry().lock();
    reg.stores
        .remove(&store.0)
        .map(|_| ())
        .ok_or(StoreError::NotFound)
}
