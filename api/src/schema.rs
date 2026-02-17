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
/// Flat — no recursive variants. `List` and `Map` come in a later layer
/// when subscriptions need nested stores.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldKind {
    Bool,
    U8,
    U32,
    U64,
    I64,
    Str,
}
