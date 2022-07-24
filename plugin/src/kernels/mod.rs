static TYPE_CONSTRAINT_T: &str = "T\0";

mod bias_add;
mod relu;

#[no_mangle]
pub extern "C" fn TF_InitKernel() {
    bias_add::init();
    relu::init();
}
