use crate::{
    bindings::{
        kernels::KernelBuilder,
        raw::{TF_OpKernelConstruction, TF_OpKernelContext, TF_TensorData, TF_FLOAT},
    },
    kernels::TYPE_CONSTRAINT_T,
    DEVICE_TYPE,
};

static RELU_KERNEL_NAME: &str = "Relu\0";
static RELU_OP_NAME: &str = "ReluOp\0";

struct ReluKernel {}

pub fn init() {
    KernelBuilder::<ReluKernel>::new(RELU_KERNEL_NAME, RELU_OP_NAME, DEVICE_TYPE)
        .constraint(TYPE_CONSTRAINT_T, TF_FLOAT)
        .create(create)
        .compute(compute)
        .delete(delete)
        .register()
}

extern "C" fn create(_construction: *mut TF_OpKernelConstruction) -> *mut ReluKernel {
    Box::into_raw(Box::new(ReluKernel {}))
}

extern "C" fn compute(_kernel: *mut ReluKernel, ctx: *mut TF_OpKernelContext) {
    let device_data = unsafe {
        &*match ctx.get_stream::<String>() {
            Ok(stream) => stream,
            Err(status) => return ctx.failure(status),
        }
    };

    eprintln!("device passed into kernel: {}", device_data);

    let input = match ctx.get_input(0) {
        Ok(input) => input,
        Err(status) => return ctx.failure(status),
    };

    let len = input.element_count() as usize;
    if len == 0 {
        return;
    }

    let dims = input.dims();
    let output = match ctx.allocate_output(
        0,
        &dims,
        (input.element_count() as u64) * (std::mem::size_of::<f32>() as u64),
    ) {
        Ok(output) => output,
        Err(status) => return ctx.failure(status),
    };

    let input_raw: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(TF_TensorData(input) as *mut f32, len) };
    let output_raw: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(TF_TensorData(output) as *mut f32, len) };

    for i in 0..len {
        output_raw[i] = match input_raw[i] > 0f32 {
            true => input_raw[i],
            false => 0f32,
        };
    }
}

unsafe extern "C" fn delete(kernel: *mut ReluKernel) {
    std::mem::drop(Box::from_raw(kernel))
}
