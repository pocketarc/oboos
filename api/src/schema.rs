//! Store schema metadata — the shape of a store's fields.
//!
//! A schema declares the set of named, typed fields a store instance holds.
//! The kernel validates every `set()` against the schema at runtime, rejecting
//! unknown fields and type mismatches before any data changes.
//!
//! Schemas are defined by implementing [`StoreSchema`] on a marker type. A
//! `#[derive(StoreSchema)]` proc-macro will generate these impls automatically
//! in a later phase — for now, manual impls work fine.

/// A store's schema: its name and the fields it contains.
///
/// Implement this on a unit struct to define a schema:
///
/// ```
/// struct CounterSchema;
///
/// impl StoreSchema for CounterSchema {
///     fn name() -> &'static str { "Counter" }
///     fn fields() -> &'static [FieldDef] {
///         &[
///             FieldDef { name: "count", kind: FieldKind::U32 },
///             FieldDef { name: "label", kind: FieldKind::Str },
///         ]
///     }
/// }
/// ```
pub trait StoreSchema {
    /// The human-readable name for this schema (for debug/logging).
    fn name() -> &'static str {
        ""
    }

    /// The fields this schema defines — name and type for each.
    fn fields() -> &'static [FieldDef];
}

/// A single field definition: a name and its expected type.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub name: &'static str,
    pub kind: FieldKind,
}

/// The type of a store field.
///
/// Scalar variants are flat values. Collection types are parameterized by
/// their element kind:
///
/// - `Queue` — consumed on read (FIFO). SET pushes, GET pops.
/// - `List` — retained on read (ordered). SET pushes, GET returns full list.
///
/// No new syscalls needed — the store schema drives the behavior.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldKind {
    Bool,
    U8,
    U32,
    U64,
    I64,
    Str,
    /// A FIFO queue whose elements must match the inner [`FieldKind`].
    ///
    /// The `&'static` reference works because schemas are always defined
    /// as static data (`fn fields() -> &'static [FieldDef]`).
    Queue(&'static FieldKind),
    /// A retained ordered collection whose elements must match the inner
    /// [`FieldKind`]. Unlike Queue, elements persist across reads — GET
    /// returns the full list, not a single element.
    List(&'static FieldKind),
    /// A structured value with named, typed fields — like a struct.
    ///
    /// Reuses [`FieldDef`] directly, so a Record's fields are defined
    /// the same way as a store's fields. Supports nesting:
    /// `List(&FieldKind::Record(&[...]))` is a list of records.
    Record(&'static [FieldDef]),
}
