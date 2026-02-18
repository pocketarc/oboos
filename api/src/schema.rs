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
pub struct FieldDef {
    pub name: &'static str,
    pub kind: FieldKind,
}

/// The type of a store field.
///
/// Scalar variants are flat values. `Queue` is the first collection type:
/// a FIFO queue parameterized by its element kind. SET on a Queue field
/// pushes one element; GET pops one. No new syscalls needed — the store
/// schema drives the behavior.
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
}
