use std::cell::Cell;

thread_local! {
    static DEBUG_MODE: Cell<bool> = const { Cell::new(false) };
    static VERBOSE_MODE: Cell<bool> = const { Cell::new(false) };
}

/// Set debug mode on or off.
pub fn set_debug_mode(enabled: bool) {
    DEBUG_MODE.with(|flag| flag.set(enabled));
}

/// Set verbose mode on or off.
pub fn set_verbose_mode(enabled: bool) {
    VERBOSE_MODE.with(|flag| flag.set(enabled));
}

pub fn is_debug_enabled() -> bool {
    DEBUG_MODE.with(|flag| flag.get())
}

pub fn is_verbose_enabled() -> bool {
    VERBOSE_MODE.with(|flag| flag.get())
}

#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        if $crate::debug::is_debug_enabled() || $crate::debug::is_verbose_enabled() {
            println!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! verbose_println {
    ($($arg:tt)*) => {
        if $crate::debug::is_verbose_enabled() {
            println!($($arg)*);
        }
    };
}
