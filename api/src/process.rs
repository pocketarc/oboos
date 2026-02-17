//! Process specification — declares resources at spawn time.
//!
//! A [`ProcessSpec`] is the "birth certificate" of a process: it declares
//! everything the process needs before it starts running. This is the
//! minimal Layer 0 skeleton — capabilities, supervision trees, and store
//! schema references all come in later layers.

use alloc::string::String;

/// Specification for spawning a new process.
///
/// Passed to the (future) process spawner to declare what the process
/// needs. The kernel validates and allocates resources based on this
/// spec before the process starts executing.
pub struct ProcessSpec {
    /// Human-readable process name (for debug/logging).
    pub name: String,
    /// Maximum heap memory this process may allocate, in bytes.
    pub memory_limit: usize,
}
