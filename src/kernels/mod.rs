static TYPE_CONSTRAINT_T: &str = "T\0";

mod relu;

#[no_mangle]
pub unsafe extern "C" fn TF_InitKernel() {
    relu::init_relu_kernel();
}
