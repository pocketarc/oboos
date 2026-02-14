//! Architecture-specific code.
//!
//! Conditionally compiles the correct module and re-exports its types.
//! The kernel uses `arch::*` everywhere — adding aarch64 later means
//! adding another pair of cfg lines and implementing the same traits.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

// Future:
// #[cfg(target_arch = "aarch64")]
// pub mod aarch64;
// #[cfg(target_arch = "aarch64")]
// pub use aarch64::*;
