use crate::{
    bindings::{kernels::*, raw::*},
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

unsafe extern "C" fn create(_construction: *mut TF_OpKernelConstruction) -> *mut ReluKernel {
    Box::into_raw(Box::new(ReluKernel {}))
}

unsafe extern "C" fn compute(_kernel: *mut ReluKernel, ctx: *mut TF_OpKernelContext) {
    let status_ptr = TF_NewStatus();
    let stream = TF_GetStream(ctx, status_ptr);
    if TF_GetCode(status_ptr) != TF_OK {
        TF_OpKernelContext_Failure(ctx, status_ptr);
        return;
    }

    let device_data = &*((*stream).stream_handle as *mut String);
    eprintln!("device passed into kernel: {}", device_data);

    let mut input_ptr = 0 as *mut TF_Tensor;
    let status_ptr = TF_NewStatus();

    TF_GetInput(ctx, 0, &mut input_ptr, status_ptr);
    if TF_GetCode(status_ptr) != TF_OK {
        TF_OpKernelContext_Failure(ctx, status_ptr);
        return;
    }

    if TF_TensorElementCount(input_ptr) == 0 {
        return;
    }

    let ndim = TF_NumDims(input_ptr);
    let mut dims: Vec<i64> = Vec::new();

    for i in 0..ndim {
        dims.push(TF_Dim(input_ptr, i));
    }

    let output_ptr = TF_AllocateOutput(
        ctx,
        0,
        TF_ExpectedOutputDataType(ctx, 0),
        dims.as_ptr(),
        dims.len() as i32,
        (TF_TensorElementCount(input_ptr) as u64) * (std::mem::size_of::<f32>() as u64),
        status_ptr,
    );

    if TF_GetCode(status_ptr) != TF_OK {
        TF_OpKernelContext_Failure(ctx, status_ptr);
        return;
    }

    let len = TF_TensorElementCount(input_ptr) as usize;
    let input_raw: &mut [f32] =
        std::slice::from_raw_parts_mut(TF_TensorData(input_ptr) as *mut f32, len);
    let output_raw: &mut [f32] =
        std::slice::from_raw_parts_mut(TF_TensorData(output_ptr) as *mut f32, len);
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
