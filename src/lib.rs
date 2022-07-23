static EMPTY_CSTR: &str = "\0";

pub static DEVICE_NAME: &str = "FAKE_CPU_DEVICE\0";
pub static DEVICE_TYPE: &str = "MY_DEVICE\0";

mod bindings;
mod kernels;
mod optimizer;
mod plugin;
