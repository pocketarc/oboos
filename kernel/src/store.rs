//! Kernel store registry — reactive state trees for IPC.
//!
//! The store is OBOOS's future IPC primitive: a shared, schema-validated
//! state tree that processes read, write, and subscribe to. This layer
//! adds field-level subscriptions with [`Waker`] integration, enabling
//! async tasks to reactively watch store fields.
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
//!
//! ## Subscriptions
//!
//! [`watch()`] returns a one-shot future that resolves the next time any
//! of the watched fields is written via [`set()`]. Use it in a loop for
//! persistent watching. When `set()` updates fields, it drains all
//! matching subscribers and wakes them after releasing the registry lock.
//!
//! ## Atomicity
//!
//! [`set()`] accepts multiple field/value pairs and writes them all under
//! a single lock acquisition. Validation is all-or-nothing: if any field
//! fails (unknown name, type mismatch), no fields are written. This
//! guarantees observers always see a consistent snapshot.
//!
//! ## Lock ordering
//!
//! Registry lock → executor lock. This is safe because the executor
//! releases its lock before polling futures (futures are removed from
//! the task map before `poll()`). The `set()` path collects wakers
//! under the registry lock, drops it, then calls `wake()` which takes
//! the executor lock — never both at once.
//!
//! ## Interrupt safety
//!
//! All public functions bracket their registry access with `cli`/`sti`
//! because `set()` calls `wake()`, and wakers push to the executor's
//! wake queue (which the PIT tick handler also touches). Same pattern
//! as [`crate::timer`] and [`crate::executor`].

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::future::Future;
use core::task::Waker;

use crate::arch::Arch;
use crate::platform::Platform;
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

/// Opaque store identifier — monotonic, never reused.
///
/// Typed `StoreHandle<S>` wraps this in a later layer to provide
/// compile-time schema guarantees. At Layer 0, callers pass raw IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StoreId(u64);

impl StoreId {
    /// Reconstruct a `StoreId` from its raw `u64` representation.
    ///
    /// Used at the syscall boundary: userspace passes the raw ID in a
    /// register, and the syscall handler converts it back to a typed ID.
    pub fn from_raw(raw: u64) -> Self {
        StoreId(raw)
    }

    /// Extract the raw `u64` value for passing across the syscall boundary.
    pub fn as_raw(self) -> u64 {
        self.0
    }
}

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

/// A registered subscriber waiting for any of its watched fields to change.
///
/// Each [`watch()`] call creates exactly one subscriber, regardless of
/// how many fields are watched. This prevents duplicate wakes when
/// [`set()`] writes multiple watched fields atomically.
struct Subscriber {
    fields: &'static [&'static str],
    waker: Waker,
}

/// A live store instance: schema metadata + current field values + subscribers.
struct StoreInstance {
    fields: &'static [FieldDef],
    #[allow(dead_code)]
    schema_name: &'static str,
    data: BTreeMap<String, Value>,
    subscribers: Vec<Subscriber>,
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
    Arch::disable_interrupts();
    let result = create_inner::<S>(defaults);
    Arch::enable_interrupts();
    result
}

fn create_inner<S: StoreSchema>(defaults: &[(&str, Value)]) -> Result<StoreId, StoreError> {
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
            // Pre-allocate subscriber slots so wakers don't hit the allocator
            // on the hot path.
            subscribers: Vec::with_capacity(8),
        },
    );
    Ok(StoreId(id))
}

/// Read a field's current value (cloned out).
pub fn get(store: StoreId, field: &str) -> Result<Value, StoreError> {
    Arch::disable_interrupts();
    let result = get_inner(store, field);
    Arch::enable_interrupts();
    result
}

/// Read a field without touching the interrupt flag.
///
/// The syscall handler already runs with IF=0 (FMASK clears it on
/// SYSCALL entry). Calling the normal `get()` would re-enable
/// interrupts via `sti` on exit — breaking the syscall return path
/// which expects IF=0 for the `sysretq` instruction.
pub(crate) fn get_no_cli(store: StoreId, field: &str) -> Result<Value, StoreError> {
    get_inner(store, field)
}

/// Look up a field's [`FieldKind`] from the schema without reading its value.
///
/// Used by the syscall handler to interpret raw bytes from userspace
/// according to the field's declared type. Runs without touching the
/// interrupt flag (same rationale as [`get_no_cli`]).
pub(crate) fn field_kind_no_cli(store: StoreId, field: &str) -> Result<FieldKind, StoreError> {
    let reg = registry().lock();
    let instance = reg.stores.get(&store.0).ok_or(StoreError::NotFound)?;
    let field_def = instance
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or(StoreError::UnknownField)?;
    Ok(field_def.kind.clone())
}

fn get_inner(store: StoreId, field: &str) -> Result<Value, StoreError> {
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

/// Write one or more field values atomically, with schema validation.
///
/// All fields are validated before any writes happen — if any field is
/// unknown or has a type mismatch, the entire call fails and nothing
/// changes. After a successful write, all subscribers watching any of
/// the written fields are woken.
///
/// # Examples
///
/// ```
/// // Single field:
/// store::set(id, &[("count", Value::U32(42))])?;
///
/// // Multiple fields atomically:
/// store::set(id, &[
///     ("uptime_s", Value::U64(10)),
///     ("free_kb", Value::U64(50000)),
/// ])?;
/// ```
pub fn set(store: StoreId, updates: &[(&str, Value)]) -> Result<(), StoreError> {
    Arch::disable_interrupts();
    let result = set_inner(store, updates);
    Arch::enable_interrupts();
    result
}

/// Write fields without touching the interrupt flag.
///
/// Same rationale as [`get_no_cli`] — the syscall path runs with IF=0
/// and must stay that way through `sysretq`.
pub(crate) fn set_no_cli(store: StoreId, updates: &[(&str, Value)]) -> Result<(), StoreError> {
    set_inner(store, updates)
}

fn set_inner(store: StoreId, updates: &[(&str, Value)]) -> Result<(), StoreError> {
    // Collect wakers under the registry lock, then fire them after
    // releasing it. This avoids holding registry + executor locks
    // simultaneously (see module-level lock ordering docs).
    let wakers = {
        let mut reg = registry().lock();
        let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

        // Validate all fields before writing any (all-or-nothing).
        for &(field, ref value) in updates {
            let field_def = instance
                .fields
                .iter()
                .find(|f| f.name == field)
                .ok_or(StoreError::UnknownField)?;

            if !value.matches(&field_def.kind) {
                return Err(StoreError::TypeMismatch);
            }
        }

        // All valid — write them all.
        for &(field, ref value) in updates {
            instance.data.insert(String::from(field), value.clone());
        }

        // Drain subscribers whose watched fields overlap with the written
        // fields. Each subscriber is a single entry (even if watching
        // multiple fields), so one wake per watch() call.
        let mut wakers = Vec::new();
        let mut i = 0;
        while i < instance.subscribers.len() {
            let sub_fields = instance.subscribers[i].fields;
            if updates.iter().any(|&(f, _)| sub_fields.contains(&f)) {
                wakers.push(instance.subscribers.swap_remove(i).waker);
            } else {
                i += 1;
            }
        }
        wakers
    }; // registry lock dropped here

    for w in wakers {
        w.wake();
    }
    Ok(())
}

/// Destroy a store instance, freeing its data.
///
/// After this call, any `get`/`set`/`destroy` with this ID returns
/// [`StoreError::NotFound`]. The ID is never reused. All active
/// subscribers are woken so their futures can detect the store is gone
/// (they'll get `NotFound` on their next poll).
pub fn destroy(store: StoreId) -> Result<(), StoreError> {
    Arch::disable_interrupts();
    let result = destroy_inner(store);
    Arch::enable_interrupts();
    result
}

fn destroy_inner(store: StoreId) -> Result<(), StoreError> {
    let wakers = {
        let mut reg = registry().lock();
        let instance = reg
            .stores
            .remove(&store.0)
            .ok_or(StoreError::NotFound)?;

        // Wake all subscribers so their futures can observe NotFound.
        instance
            .subscribers
            .into_iter()
            .map(|s| s.waker)
            .collect::<Vec<_>>()
    }; // registry lock dropped here

    for w in wakers {
        w.wake();
    }
    Ok(())
}

/// Register a subscriber for one or more fields (private).
///
/// Validates that every field exists in the schema, then pushes a
/// single [`Subscriber`] covering all watched fields. Called from
/// [`watch()`]'s first poll.
fn subscribe(store: StoreId, fields: &'static [&'static str], waker: Waker) -> Result<(), StoreError> {
    let mut reg = registry().lock();
    let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

    for &field in fields {
        if !instance.fields.iter().any(|f| f.name == field) {
            return Err(StoreError::UnknownField);
        }
    }

    instance.subscribers.push(Subscriber { fields, waker });
    Ok(())
}

/// Watch one or more store fields for changes.
///
/// Returns a one-shot future that resolves the next time [`set()`]
/// writes to any of the watched fields. Use in a loop for persistent
/// watching:
///
/// ```
/// loop {
///     store::watch(id, &["uptime_s", "free_kb"]).await?;
///     let uptime = store::get(id, "uptime_s")?;
///     let free = store::get(id, "free_kb")?;
///     // react to new values
/// }
/// ```
///
/// If the store is destroyed while watching, the future resolves with
/// [`StoreError::NotFound`].
pub fn watch(
    store: StoreId,
    fields: &'static [&'static str],
) -> impl Future<Output = Result<(), StoreError>> + Send {
    let mut subscribed = false;

    core::future::poll_fn(move |cx| {
        if !subscribed {
            // First poll: register wakers for all fields and return Pending.
            Arch::disable_interrupts();
            match subscribe(store, fields, cx.waker().clone()) {
                Ok(()) => {
                    subscribed = true;
                    Arch::enable_interrupts();
                    return core::task::Poll::Pending;
                }
                Err(e) => {
                    Arch::enable_interrupts();
                    return core::task::Poll::Ready(Err(e));
                }
            }
        }

        // Woken by set() or destroy(). Check the store still exists
        // so watchers can detect destruction.
        Arch::disable_interrupts();
        let exists = registry().lock().stores.contains_key(&store.0);
        Arch::enable_interrupts();

        if exists {
            core::task::Poll::Ready(Ok(()))
        } else {
            core::task::Poll::Ready(Err(StoreError::NotFound))
        }
    })
}
