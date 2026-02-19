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
use core::future::Future;
use core::task::Waker;

use crate::arch::Arch;
use crate::platform::Platform;
use oboos_api::{FieldDef, FieldKind, StoreSchema, Value};

pub use oboos_api::StoreError;

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

/// A registered subscriber waiting for any of its watched fields to change.
///
/// Each [`watch()`] call creates exactly one subscriber, regardless of
/// how many fields are watched. This prevents duplicate wakes when
/// [`set()`] writes multiple watched fields atomically.
///
/// `fields` is a `Vec` so both kernel-side watchers (passing static
/// slices that `.to_vec()`) and the syscall path (constructing a
/// single-field vec from a dynamically-resolved `&'static str`) can
/// create subscribers without lifetime gymnastics.
struct Subscriber {
    fields: Vec<&'static str>,
    waker: Waker,
}

/// A persistent watcher — not drained on fire, only removed explicitly.
///
/// Unlike one-shot [`Subscriber`]s which are consumed when they fire,
/// persistent watchers stay registered across multiple writes. The waker
/// is cloned (not consumed) on each fire. Used by the SYS_SUBSCRIBE
/// syscall to give userspace a stream-like API.
struct PersistentWatcher {
    field: &'static str,
    waker: Waker,
    /// Unique ID for removal by SYS_UNSUBSCRIBE.
    id: u64,
}

/// A live store instance: schema metadata + current field values + subscribers.
struct StoreInstance {
    fields: &'static [FieldDef],
    #[allow(dead_code)]
    schema_name: &'static str,
    data: BTreeMap<String, Value>,
    subscribers: Vec<Subscriber>,
    watchers: Vec<PersistentWatcher>,
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
            watchers: Vec::new(),
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
///     ("uptime_ms", Value::U64(10_042)),
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

        // Drain one-shot subscribers whose watched fields overlap with
        // the written fields.
        let mut wakers = Vec::new();
        let mut i = 0;
        while i < instance.subscribers.len() {
            let sub_fields = &instance.subscribers[i].fields;
            if updates.iter().any(|&(f, _)| sub_fields.contains(&f)) {
                wakers.push(instance.subscribers.swap_remove(i).waker);
            } else {
                i += 1;
            }
        }

        // Clone wakers from persistent watchers (NOT drained).
        for watcher in &instance.watchers {
            if updates.iter().any(|&(f, _)| f == watcher.field) {
                wakers.push(watcher.waker.clone());
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

        // Wake all one-shot subscribers and persistent watchers so their
        // futures can observe NotFound.
        let mut wakers: Vec<Waker> = instance
            .subscribers
            .into_iter()
            .map(|s| s.waker)
            .collect();
        for watcher in &instance.watchers {
            wakers.push(watcher.waker.clone());
        }
        wakers
    }; // registry lock dropped here

    for w in wakers {
        w.wake();
    }
    Ok(())
}

// ————————————————————————————————————————————————————————————————————————————
// Queue operations
// ————————————————————————————————————————————————————————————————————————————

/// Push a value onto a Queue field.
///
/// Validates that the field is a `Queue(inner)` and that the value matches
/// the inner kind. Appends to the back of the deque, then fires wakers on
/// any subscribers watching this field — same drain-and-wake pattern as
/// [`set()`].
pub fn push(store: StoreId, field: &str, value: Value) -> Result<(), StoreError> {
    Arch::disable_interrupts();
    let result = push_inner(store, field, value);
    Arch::enable_interrupts();
    result
}

/// Push without touching the interrupt flag (syscall path).
pub(crate) fn push_no_cli(store: StoreId, field: &str, value: Value) -> Result<(), StoreError> {
    push_inner(store, field, value)
}

fn push_inner(store: StoreId, field: &str, value: Value) -> Result<(), StoreError> {
    let wakers = {
        let mut reg = registry().lock();
        let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

        let field_def = instance
            .fields
            .iter()
            .find(|f| f.name == field)
            .ok_or(StoreError::UnknownField)?;

        let inner_kind = match &field_def.kind {
            FieldKind::Queue(inner) => *inner,
            _ => return Err(StoreError::TypeMismatch),
        };

        if !value.matches(inner_kind) {
            return Err(StoreError::TypeMismatch);
        }

        // Append to the queue.
        match instance.data.get_mut(field) {
            Some(Value::Queue(deque)) => deque.push_back(value),
            _ => return Err(StoreError::TypeMismatch),
        }

        // Wake one-shot subscribers watching this field.
        let mut wakers = Vec::new();
        let mut i = 0;
        while i < instance.subscribers.len() {
            if instance.subscribers[i].fields.iter().any(|&f| f == field) {
                wakers.push(instance.subscribers.swap_remove(i).waker);
            } else {
                i += 1;
            }
        }

        // Clone wakers from persistent watchers (NOT drained).
        for watcher in &instance.watchers {
            if watcher.field == field {
                wakers.push(watcher.waker.clone());
            }
        }

        wakers
    }; // registry lock dropped here

    for w in wakers {
        w.wake();
    }
    Ok(())
}

/// Pop the front element from a Queue field.
///
/// Returns `Ok(Some(value))` if the queue had an element, `Ok(None)` if
/// empty. Does not fire wakers — this is a consumer-side operation.
pub fn pop(store: StoreId, field: &str) -> Result<Option<Value>, StoreError> {
    Arch::disable_interrupts();
    let result = pop_inner(store, field);
    Arch::enable_interrupts();
    result
}

/// Pop without touching the interrupt flag (syscall path).
pub(crate) fn pop_no_cli(store: StoreId, field: &str) -> Result<Option<Value>, StoreError> {
    pop_inner(store, field)
}

fn pop_inner(store: StoreId, field: &str) -> Result<Option<Value>, StoreError> {
    let mut reg = registry().lock();
    let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

    let field_def = instance
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or(StoreError::UnknownField)?;

    if !matches!(field_def.kind, FieldKind::Queue(_)) {
        return Err(StoreError::TypeMismatch);
    }

    match instance.data.get_mut(field) {
        Some(Value::Queue(deque)) => Ok(deque.pop_front()),
        _ => Err(StoreError::TypeMismatch),
    }
}

/// Drain all elements from a Queue field, returning them as a Vec.
///
/// Returns an empty Vec if the queue was already empty. Does not fire
/// wakers — this is a consumer-side operation.
pub fn drain(store: StoreId, field: &str) -> Result<Vec<Value>, StoreError> {
    Arch::disable_interrupts();
    let result = drain_inner(store, field);
    Arch::enable_interrupts();
    result
}

/// Drain without touching the interrupt flag (syscall path).
#[allow(dead_code)]
pub(crate) fn drain_no_cli(store: StoreId, field: &str) -> Result<Vec<Value>, StoreError> {
    drain_inner(store, field)
}

fn drain_inner(store: StoreId, field: &str) -> Result<Vec<Value>, StoreError> {
    let mut reg = registry().lock();
    let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

    let field_def = instance
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or(StoreError::UnknownField)?;

    if !matches!(field_def.kind, FieldKind::Queue(_)) {
        return Err(StoreError::TypeMismatch);
    }

    match instance.data.get_mut(field) {
        Some(Value::Queue(deque)) => Ok(deque.drain(..).collect()),
        _ => Err(StoreError::TypeMismatch),
    }
}

/// Register a subscriber for one or more fields (private).
///
/// Validates that every field exists in the schema, then pushes a
/// single [`Subscriber`] covering all watched fields. Called from
/// [`watch()`]'s first poll.
fn subscribe(store: StoreId, fields: &[&'static str], waker: Waker) -> Result<(), StoreError> {
    let mut reg = registry().lock();
    let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

    for &field in fields {
        if !instance.fields.iter().any(|f| f.name == field) {
            return Err(StoreError::UnknownField);
        }
    }

    instance.subscribers.push(Subscriber { fields: fields.to_vec(), waker });
    Ok(())
}

/// Register a persistent watcher for a single field, without touching
/// the interrupt flag. Used by SYS_SUBSCRIBE.
///
/// Returns `Ok(true)` if the field is a Queue with non-empty data,
/// signalling the caller should fire the waker immediately to prevent
/// missed data. Returns `Ok(false)` otherwise.
pub(crate) fn add_watcher_no_cli(
    store: StoreId,
    field: &str,
    waker: Waker,
    watcher_id: u64,
) -> Result<bool, StoreError> {
    let mut reg = registry().lock();
    let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

    let field_def = instance
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or(StoreError::UnknownField)?;

    // Use the schema's &'static str so the watcher's lifetime is sound.
    let static_name: &'static str = field_def.name;

    // Check if the field is a non-empty queue — caller should fire
    // immediately so userspace doesn't miss already-buffered data.
    let fire_immediately = matches!(&field_def.kind, FieldKind::Queue(_))
        && matches!(instance.data.get(field), Some(Value::Queue(q)) if !q.is_empty());

    instance.watchers.push(PersistentWatcher {
        field: static_name,
        waker,
        id: watcher_id,
    });

    Ok(fire_immediately)
}

/// Remove a persistent watcher by its ID, without touching the
/// interrupt flag. Used by SYS_UNSUBSCRIBE.
pub(crate) fn remove_watcher_no_cli(
    store: StoreId,
    watcher_id: u64,
) -> Result<(), StoreError> {
    let mut reg = registry().lock();
    let instance = reg.stores.get_mut(&store.0).ok_or(StoreError::NotFound)?;

    if let Some(pos) = instance.watchers.iter().position(|w| w.id == watcher_id) {
        instance.watchers.swap_remove(pos);
    }
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
///     store::watch(id, &["uptime_ms", "free_kb"]).await?;
///     let uptime = store::get(id, "uptime_ms")?;
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
