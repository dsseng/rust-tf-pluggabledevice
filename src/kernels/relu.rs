use crate::{bindings::raw::*, kernels::TYPE_CONSTRAINT_T, DEVICE_TYPE};

static RELU_KERNEL_NAME: &str = "Relu\0";
static RELU_OP_NAME: &str = "ReluOp\0";

pub unsafe fn init_relu_kernel() {
    let builder = TF_NewKernelBuilder(
        RELU_KERNEL_NAME.as_ptr() as *const i8,
        DEVICE_TYPE.as_ptr() as *const i8,
        None,
        Some(relu_kernel_compute),
        None,
    );

    let status = TF_NewStatus();
    TF_KernelBuilder_TypeConstraint(
        builder,
        TYPE_CONSTRAINT_T.as_ptr() as *const i8,
        TF_FLOAT,
        status,
    );
    if TF_OK != TF_GetCode(status) {
        eprintln!("Error while registering relu kernel with attribute T");
        return;
    }

    TF_RegisterKernelBuilder(RELU_OP_NAME.as_ptr() as *const i8, builder, status);
    if TF_OK != TF_GetCode(status) {
        eprintln!("Error while registering relu kernel");
        return;
    }
}

pub unsafe extern "C" fn relu_kernel_compute(
    _kernel: *mut ::std::os::raw::c_void,
    ctx: *mut TF_OpKernelContext,
) {
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
