//! OBOOS userspace runtime library.
//!
//! Provides the syscall ABI, high-level wrappers, an async executor, and a
//! panic handler for all Ring 3 programs. Any code that a second userspace
//! binary would need to copy-paste belongs here instead of in the application.
//!
//! ## Syscall interface
//!
//! Six syscalls express all kernel interaction through the store:
//!
//! | # | Name            | Args                                                    | Returns            |
//! |---|-----------------|----------------------------------------------------------|--------------------|
//! | 0 | SYS_STORE_GET   | store_id, fields_ptr, fields_len, out_ptr, out_len      | bytes written      |
//! | 1 | SYS_STORE_SET   | store_id, buf_ptr, buf_len, 0, 0                        | 0 on success       |
//! | 2 | SYS_SUBSCRIBE   | store_id, field_ptr, field_len, 0, 0                    | sub_id (0-63)      |
//! | 3 | SYS_UNSUBSCRIBE | sub_id, 0, 0, 0, 0                                     | 0 on success       |
//! | 4 | SYS_YIELD       | 0, 0, 0, 0, 0                                          | fired bitmask      |
//! | 5 | SYS_STORE_MUTATE| store_id, mutation_id, payload_ptr, payload_len, 0      | mutation-specific   |
//!
//! GET and SET use packed buffers with u16 length prefixes for multi-field
//! operations. All use 5 arguments passed in RDI, RSI, RDX, R10, R8
//! (Linux convention). SYSCALL clobbers RCX (saves RIP) and R11 (saves RFLAGS).
//!
//! ## Error handling
//!
//! All fallible functions return `Result<T, StoreError>`. The [`StoreError`]
//! enum is shared with the kernel via the `oboos-api` crate. Raw `u64` error
//! codes from syscalls are converted via [`StoreError::from_raw`].
//!
//! ## Type-generic store access
//!
//! [`store_get`] and [`store_set`] use the [`FromStoreBytes`] and
//! [`IntoStoreBytes`] traits to support multiple value types through a
//! single generic function. The compiler dispatches at the call site:
//!
//! ```
//! store_set(id, "counter", 42u64)?;
//! let val: u64 = store_get(id, "counter")?;
//! let byte: Option<u8> = store_get(CONSOLE, "input")?;
//! ```
//!
//! ## Well-known store IDs
//!
//! Bit 63 flags well-known stores resolved per-process by the kernel:
//! - [`PROCESS`] — current process's lifecycle store (pid, status, exit_code)
//! - [`CONSOLE`] — serial console device store (output + input queues)

#![no_std]

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::future::Future;
use core::pin::Pin;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

pub use oboos_api::{StoreError, is_error, ERR_THRESHOLD};

// ————————————————————————————————————————————————————————————————————————————
// Syscall numbers — must match kernel/src/arch/x86_64/syscall.rs
// ————————————————————————————————————————————————————————————————————————————

pub const SYS_STORE_GET: u64 = 0;
pub const SYS_STORE_SET: u64 = 1;
pub const SYS_SUBSCRIBE: u64 = 2;
pub const SYS_UNSUBSCRIBE: u64 = 3;
pub const SYS_YIELD: u64 = 4;
pub const SYS_STORE_MUTATE: u64 = 5;

// ————————————————————————————————————————————————————————————————————————————
// Well-known store IDs
// ————————————————————————————————————————————————————————————————————————————

/// The current process's lifecycle store. The kernel resolves this to the
/// process store created by `process::spawn()`, containing fields like
/// `pid` (U64), `name` (Str), `status` (Str), and `exit_code` (U64).
pub const PROCESS: u64 = 1 << 63;

/// The serial console device store. Contains `"output"` (Queue(Str)) for
/// writing and `"input"` (Queue(U8)) for reading keyboard input.
pub const CONSOLE: u64 = (1 << 63) | 1;

// ————————————————————————————————————————————————————————————————————————————
// Raw syscall wrapper
// ————————————————————————————————————————————————————————————————————————————

/// Syscall with 5 arguments — the only raw syscall needed, since all
/// syscalls take exactly 5 args (unused args are 0).
///
/// Register mapping follows the Linux SYSCALL convention:
/// RAX = number, RDI = arg1, RSI = arg2, RDX = arg3, R10 = arg4, R8 = arg5.
#[inline(always)]
pub fn syscall5(number: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> u64 {
    let ret: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") number,
            in("rdi") arg1,
            in("rsi") arg2,
            in("rdx") arg3,
            in("r10") arg4,
            in("r8") arg5,
            lateout("rax") ret,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    ret
}

// ————————————————————————————————————————————————————————————————————————————
// Packed buffer encoding/decoding helpers
// ————————————————————————————————————————————————————————————————————————————

/// Encode a single field name into a GET input buffer.
///
/// Writes `[u16 name_len (LE), name bytes]` starting at `buf[off]`.
/// Advances `off` past the written data.
fn encode_field(buf: &mut [u8], off: &mut usize, name: &str) {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as u16;
    buf[*off..*off + 2].copy_from_slice(&name_len.to_le_bytes());
    *off += 2;
    buf[*off..*off + name_bytes.len()].copy_from_slice(name_bytes);
    *off += name_bytes.len();
}

/// Encode a field name + value into a SET buffer.
///
/// Writes `[u16 name_len, name, u16 value_len, value]` starting at `buf[off]`.
/// Advances `off` past the written data.
fn encode_pair(buf: &mut [u8], off: &mut usize, name: &str, value: &[u8]) {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as u16;
    buf[*off..*off + 2].copy_from_slice(&name_len.to_le_bytes());
    *off += 2;
    buf[*off..*off + name_bytes.len()].copy_from_slice(name_bytes);
    *off += name_bytes.len();

    let val_len = value.len() as u16;
    buf[*off..*off + 2].copy_from_slice(&val_len.to_le_bytes());
    *off += 2;
    buf[*off..*off + value.len()].copy_from_slice(value);
    *off += value.len();
}

/// Decode one value from a GET output buffer.
///
/// Reads `[u16 val_len, val bytes]` starting at `buf[off]`.
/// Advances `off` past the read data. Returns the value bytes.
fn decode_value<'a>(buf: &'a [u8], off: &mut usize) -> &'a [u8] {
    let val_len = u16::from_le_bytes([buf[*off], buf[*off + 1]]) as usize;
    *off += 2;
    let val = &buf[*off..*off + val_len];
    *off += val_len;
    val
}

/// Convert a raw syscall return value into a `Result`.
/// Returns `Ok(val)` if not an error, `Err(StoreError)` otherwise.
fn check_error(raw: u64) -> Result<u64, StoreError> {
    if is_error(raw) {
        Err(StoreError::from_raw(raw).unwrap_or(StoreError::InvalidArg))
    } else {
        Ok(raw)
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Type-generic store traits
// ————————————————————————————————————————————————————————————————————————————

/// Types that can be serialized into a store SET buffer.
pub trait IntoStoreBytes {
    /// Write this value's bytes into `buf` and return the number of bytes written.
    fn write_to(&self, buf: &mut [u8]) -> usize;
}

impl IntoStoreBytes for u64 {
    fn write_to(&self, buf: &mut [u8]) -> usize {
        let bytes = self.to_ne_bytes();
        buf[..8].copy_from_slice(&bytes);
        8
    }
}

impl IntoStoreBytes for u8 {
    fn write_to(&self, buf: &mut [u8]) -> usize {
        buf[0] = *self;
        1
    }
}

impl IntoStoreBytes for &str {
    fn write_to(&self, buf: &mut [u8]) -> usize {
        let bytes = self.as_bytes();
        buf[..bytes.len()].copy_from_slice(bytes);
        bytes.len()
    }
}

impl IntoStoreBytes for bool {
    fn write_to(&self, buf: &mut [u8]) -> usize {
        buf[0] = *self as u8;
        1
    }
}

/// Types that can be deserialized from a store GET response.
pub trait FromStoreBytes: Sized {
    /// Interpret the raw bytes from a GET response value.
    /// Returns `None` if the bytes don't match the expected format.
    fn from_bytes(bytes: &[u8]) -> Option<Self>;

    /// Whether this value represents "no data available" (e.g. an empty
    /// queue pop). Used by [`WatchFuture`] to decide whether to keep
    /// the READY_MASK bit set for draining multi-item queues.
    fn is_empty(&self) -> bool { false }
}

impl FromStoreBytes for u64 {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 8 { return None; }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Some(u64::from_ne_bytes(arr))
    }
}

/// `Option<u8>` represents a Queue(U8) pop: empty queue returns `None`,
/// otherwise `Some(byte)`.
impl FromStoreBytes for Option<u8> {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            Some(None)
        } else {
            Some(Some(bytes[0]))
        }
    }

    fn is_empty(&self) -> bool {
        self.is_none()
    }
}

// ————————————————————————————————————————————————————————————————————————————
// High-level syscall wrappers
// ————————————————————————————————————————————————————————————————————————————

/// Read a typed value from a store field via SYS_STORE_GET.
///
/// The return type is inferred from context via [`FromStoreBytes`]:
///
/// ```
/// let counter: u64 = store_get(id, "counter")?;
/// let byte: Option<u8> = store_get(CONSOLE, "input")?;
/// ```
pub fn store_get<T: FromStoreBytes>(store_id: u64, field: &str) -> Result<T, StoreError> {
    let mut fields = [0u8; 64];
    let mut off = 0;
    encode_field(&mut fields, &mut off, field);

    let mut out = [0u8; 1024];
    let ret = syscall5(
        SYS_STORE_GET,
        store_id,
        fields.as_ptr() as u64, off as u64,
        out.as_mut_ptr() as u64, out.len() as u64,
    );
    check_error(ret)?;

    let mut decode_off = 0;
    let val = decode_value(&out[..ret as usize], &mut decode_off);
    T::from_bytes(val).ok_or(StoreError::TypeMismatch)
}

/// Read a string value from a store field into a caller-provided buffer.
///
/// Returns the number of bytes written. Separate from [`store_get`]
/// because string GET needs a caller-provided buffer (can't return
/// owned data without alloc).
pub fn store_get_str(store_id: u64, field: &str, out: &mut [u8]) -> Result<usize, StoreError> {
    let mut fields = [0u8; 64];
    let mut off = 0;
    encode_field(&mut fields, &mut off, field);

    let total_out_len = out.len() + 2;
    let mut tmp = [0u8; 1024];
    let buf_len = total_out_len.min(tmp.len());

    let ret = syscall5(
        SYS_STORE_GET,
        store_id,
        fields.as_ptr() as u64, off as u64,
        tmp.as_mut_ptr() as u64, buf_len as u64,
    );
    check_error(ret)?;
    if ret < 2 { return Ok(0); }

    let mut decode_off = 0;
    let val = decode_value(&tmp[..ret as usize], &mut decode_off);
    let copy_len = val.len().min(out.len());
    out[..copy_len].copy_from_slice(&val[..copy_len]);
    Ok(copy_len)
}

/// Write a typed value to a store field via SYS_STORE_SET.
///
/// The value type is inferred from context via [`IntoStoreBytes`]:
///
/// ```
/// store_set(id, "counter", 42u64)?;
/// store_set(CONSOLE, "output", "hello\n")?;
/// ```
pub fn store_set(store_id: u64, field: &str, value: impl IntoStoreBytes) -> Result<(), StoreError> {
    let mut val_buf = [0u8; 256];
    let val_len = value.write_to(&mut val_buf);

    let mut buf = [0u8; 512];
    let mut off = 0;
    encode_pair(&mut buf, &mut off, field, &val_buf[..val_len]);

    let ret = syscall5(SYS_STORE_SET, store_id, buf.as_ptr() as u64, off as u64, 0, 0);
    check_error(ret)?;
    Ok(())
}

// ————————————————————————————————————————————————————————————————————————————
// MUTATE syscall helpers
// ————————————————————————————————————————————————————————————————————————————

/// Request `pages` heap pages from the kernel via the MapHeap mutation.
///
/// Returns the virtual address of the start of the newly mapped region,
/// or an error code (check with [`is_error`]).
pub fn sys_mutate_map_heap(pages: u64) -> u64 {
    let payload = pages.to_ne_bytes();
    syscall5(
        SYS_STORE_MUTATE,
        PROCESS,
        oboos_api::PROCESS_MUTATE_MAP_HEAP as u64,
        payload.as_ptr() as u64,
        8,
        0,
    )
}

// ————————————————————————————————————————————————————————————————————————————
// Global heap allocator — grows via MapHeap mutation
// ————————————————————————————————————————————————————————————————————————————

/// Userspace heap allocator that uses the kernel's MapHeap mutation to grow.
///
/// Wraps `linked_list_allocator::Heap` behind a spinlock. On OOM, requests
/// more pages from the kernel (minimum 4 pages = 16 KiB per grow) and
/// extends the heap. The kernel maps the pages into the process's heap
/// region starting at 0x0100_0000.
struct OboosAllocator {
    inner: spin::Mutex<linked_list_allocator::Heap>,
}

impl OboosAllocator {
    const fn new() -> Self {
        OboosAllocator {
            inner: spin::Mutex::new(linked_list_allocator::Heap::empty()),
        }
    }
}

unsafe impl GlobalAlloc for OboosAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut heap = self.inner.lock();
        match heap.allocate_first_fit(layout) {
            Ok(ptr) => ptr.as_ptr(),
            Err(_) => {
                // OOM — ask the kernel for more pages.
                let needed = layout.size().max(layout.align());
                let pages = ((needed + 4095) / 4096).max(4); // min 16 KiB per grow
                let addr = sys_mutate_map_heap(pages as u64);
                if is_error(addr) {
                    return core::ptr::null_mut();
                }

                if heap.size() == 0 {
                    // First heap allocation — initialize the heap.
                    unsafe { heap.init(addr as *mut u8, pages * 4096); }
                } else {
                    // Extend the existing heap with the new pages.
                    unsafe { heap.extend(pages * 4096); }
                }

                heap.allocate_first_fit(layout)
                    .map_or(core::ptr::null_mut(), |p| p.as_ptr())
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            self.inner.lock().deallocate(NonNull::new_unchecked(ptr), layout);
        }
    }
}

#[global_allocator]
static ALLOCATOR: OboosAllocator = OboosAllocator::new();

/// Write a string to the serial console.
///
/// Pushes a string onto the `"output"` queue on the [`CONSOLE`] store.
/// The kernel's async console driver drains the queue to serial.
/// Fire-and-forget — ignores errors for convenience.
pub fn write(msg: &str) {
    let _ = store_set(CONSOLE, "output", msg);
}

/// Get the current process's PID.
///
/// Reads the `"pid"` field from the [`PROCESS`] well-known store.
pub fn getpid() -> u64 {
    store_get::<u64>(PROCESS, "pid").unwrap_or(0)
}

/// Exit the program with an exit code, returning control to the kernel.
///
/// Issues a MUTATE(PROCESS, Exit, code) syscall. The kernel's process store
/// reducer sets status="exited" and exit_code, then triggers the longjmp
/// back to whoever called `jump_to_ring3`. The syscall never returns.
pub fn exit(code: u64) -> ! {
    let payload = code.to_ne_bytes();
    syscall5(
        SYS_STORE_MUTATE,
        PROCESS,
        oboos_api::PROCESS_MUTATE_EXIT as u64,
        payload.as_ptr() as u64,
        8,
        0,
    );
    loop {}
}

// ————————————————————————————————————————————————————————————————————————————
// Subscription syscall wrappers
// ————————————————————————————————————————————————————————————————————————————

/// Subscribe to a store field for persistent notifications.
///
/// Returns a subscription ID (0-63) that can be used with [`unsubscribe`]
/// and appears as a bit in the bitmask returned by [`sys_yield`].
pub fn subscribe(store_id: u64, field: &str) -> Result<u64, StoreError> {
    let ret = syscall5(
        SYS_SUBSCRIBE,
        store_id,
        field.as_ptr() as u64, field.len() as u64,
        0, 0,
    );
    check_error(ret)
}

/// Remove a subscription by its ID.
pub fn unsubscribe(sub_id: u64) -> Result<(), StoreError> {
    let ret = syscall5(SYS_UNSUBSCRIBE, sub_id, 0, 0, 0, 0);
    check_error(ret)?;
    Ok(())
}

/// Yield to the kernel until at least one subscription fires.
///
/// Returns a bitmask where bit N is set if subscription N fired.
pub fn sys_yield() -> u64 {
    syscall5(SYS_YIELD, 0, 0, 0, 0, 0)
}

// ————————————————————————————————————————————————————————————————————————————
// Userspace async executor
// ————————————————————————————————————————————————————————————————————————————

/// Bitmask of subscription IDs that have fired. Updated by [`block_on`]
/// after each [`sys_yield`] call, consumed by [`WatchFuture::poll`].
static READY_MASK: AtomicU64 = AtomicU64::new(0);

/// No-op waker vtable. We always re-poll after YIELD returns, so wakers
/// don't need to do anything.
static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |data| RawWaker::new(data, &NOOP_VTABLE), // clone
    |_| {},                                     // wake
    |_| {},                                     // wake_by_ref
    |_| {},                                     // drop
);

fn noop_waker() -> Waker {
    let raw = RawWaker::new(core::ptr::null(), &NOOP_VTABLE);
    unsafe { Waker::from_raw(raw) }
}

/// Run a future to completion, yielding to the kernel between polls.
///
/// This is the userspace async executor. Each time the future returns
/// `Pending`, we call [`sys_yield`] to sleep until a subscription fires,
/// then merge the returned bitmask into [`READY_MASK`] so [`WatchFuture`]s
/// can check their bits on the next poll.
pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = core::pin::pin!(f);
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                let mask = sys_yield();
                READY_MASK.fetch_or(mask, Ordering::SeqCst);
            }
        }
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Watcher — persistent field subscription as an async stream
// ————————————————————————————————————————————————————————————————————————————

/// A persistent subscription to a store field.
///
/// Created by [`watch`]. Each call to [`next`](Watcher::next) returns a
/// future that resolves when the subscription fires and a GET succeeds.
/// Dropping the `Watcher` automatically unsubscribes.
pub struct Watcher {
    sub_id: u64,
    store_id: u64,
    field: &'static str,
}

impl Watcher {
    /// Await the next value from the watched field.
    ///
    /// Returns `Some(value)` when a new value is available, or `None`
    /// for spurious wakes (e.g. queue was empty). The return type is
    /// inferred from context via [`FromStoreBytes`].
    pub fn next<T: FromStoreBytes>(&mut self) -> WatchFuture<'_, T> {
        WatchFuture {
            watcher: self,
            _marker: core::marker::PhantomData,
        }
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        let _ = unsubscribe(self.sub_id);
    }
}

/// Future returned by [`Watcher::next`].
///
/// Checks [`READY_MASK`] for the subscription's bit. When set, clears it
/// and calls [`store_get`] to retrieve the value. If the GET succeeds,
/// re-sets the bit so the next [`Watcher::next`] call also tries a GET
/// immediately — this drains queues with multiple items without waiting
/// for a new subscription fire between each pop.
pub struct WatchFuture<'a, T> {
    watcher: &'a mut Watcher,
    _marker: core::marker::PhantomData<T>,
}

impl<T: FromStoreBytes> Future for WatchFuture<'_, T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let bit = 1u64 << self.watcher.sub_id;
        let old_mask = READY_MASK.fetch_and(!bit, Ordering::SeqCst);
        if old_mask & bit != 0 {
            match store_get::<T>(self.watcher.store_id, self.watcher.field) {
                Ok(val) if !val.is_empty() => {
                    // Re-set the bit so the next next() call also tries
                    // a GET immediately. This drains multi-item queues
                    // without waiting for a new push between each pop.
                    READY_MASK.fetch_or(bit, Ordering::SeqCst);
                    Poll::Ready(Some(val))
                }
                // GET returned "empty" (e.g. queue drained) or failed.
                // Don't re-set the bit; wait for the next subscription fire.
                _ => Poll::Pending,
            }
        } else {
            Poll::Pending
        }
    }
}

/// Create a persistent subscription to a store field.
///
/// Returns a [`Watcher`] whose [`next`](Watcher::next) method yields
/// futures that resolve when the field is written.
///
/// ```
/// let mut input = watch(CONSOLE, "input");
/// while let Some(byte) = input.next::<Option<u8>>().await {
///     // process byte
/// }
/// ```
pub fn watch(store_id: u64, field: &'static str) -> Watcher {
    let sub_id = subscribe(store_id, field).expect("subscribe failed");
    Watcher { sub_id, store_id, field }
}

// ————————————————————————————————————————————————————————————————————————————
// Decimal formatting (no alloc, no format!)
// ————————————————————————————————————————————————————————————————————————————

/// Write a u64 as decimal digits into a byte buffer. Returns bytes written.
pub fn write_u64(buf: &mut [u8], val: u64) -> usize {
    if val == 0 {
        buf[0] = b'0';
        return 1;
    }

    let mut digits = [0u8; 20];
    let mut n = val;
    let mut len = 0;
    while n > 0 {
        digits[len] = b'0' + (n % 10) as u8;
        n /= 10;
        len += 1;
    }

    for i in (0..len).rev() {
        buf[len - 1 - i] = digits[i];
    }
    len
}

// ————————————————————————————————————————————————————————————————————————————
// Panic handler
// ————————————————————————————————————————————————————————————————————————————

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    write("!!! USERSPACE PANIC !!!\n");

    if let Some(location) = info.location() {
        write(location.file());
        write(":");
        let mut buf = [0u8; 20];
        let len = write_u64(&mut buf, location.line() as u64);
        let s = unsafe { core::str::from_utf8_unchecked(&buf[..len]) };
        write(s);
        write("\n");
    }

    exit(1);
}
