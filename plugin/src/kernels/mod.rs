static TYPE_CONSTRAINT_T: &str = "T\0";

mod relu;

#[no_mangle]
pub extern "C" fn TF_InitKernel() {
    relu::init();
}
