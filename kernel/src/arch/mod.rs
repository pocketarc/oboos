/*
 * Conditionally compiles the correct architecture module.
 * The kernel uses `arch::*` everywhere — it never knows which
 * architecture is underneath. Adding aarch64 later means adding
 * another pair of cfg lines and implementing the same types.
 */

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

// Future:
// #[cfg(target_arch = "aarch64")]
// pub mod aarch64;
// #[cfg(target_arch = "aarch64")]
// pub use aarch64::*;
