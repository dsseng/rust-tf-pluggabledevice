use crate::{
    bindings::{
        compute::offset_from_tensor_coordinates,
        kernels::KernelBuilder,
        raw::{TF_OpKernelConstruction, TF_OpKernelContext, TF_TensorData, TF_FLOAT},
    },
    kernels::TYPE_CONSTRAINT_T,
    DEVICE_TYPE,
};

static BIAS_ADD_KERNEL_NAME: &str = "BiasAdd\0";
static BIAS_ADD_OP_NAME: &str = "BiasAddOp\0";

struct BiasAddKernel {
    format: String,
}

pub fn init() {
    KernelBuilder::<BiasAddKernel>::new(BIAS_ADD_KERNEL_NAME, BIAS_ADD_OP_NAME, DEVICE_TYPE)
        .constraint(TYPE_CONSTRAINT_T, TF_FLOAT)
        .create(create)
        .compute(compute)
        .delete(delete)
        .register()
}

unsafe extern "C" fn create(construction: *mut TF_OpKernelConstruction) -> *mut BiasAddKernel {
    Box::into_raw(Box::new(BiasAddKernel {
        format: construction
            .get_attr_string("data_format\0")
            .expect("Failed to get format"),
    }))
}

unsafe extern "C" fn compute(kernel: *mut BiasAddKernel, ctx: *mut TF_OpKernelContext) {
    let device_data = &*match ctx.get_stream::<String>() {
        Ok(stream) => stream,
        Err(status) => return ctx.failure(status),
    };

    let format = (*kernel).format.clone();
    eprintln!("device: {}, format: {:#?}", device_data, format);

    let input = match ctx.get_input(0) {
        Ok(input) => input,
        Err(status) => return ctx.failure(status),
    };
    let input_dims = input.dims_named(&format);

    let bias = match ctx.get_input(1) {
        Ok(input) => input,
        Err(status) => return ctx.failure(status),
    };
    let bias_dims = bias.dims();

    assert!(bias_dims.len() == 1);
    assert!(*bias_dims.first().unwrap() == input_dims.c);

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
        std::slice::from_raw_parts_mut(TF_TensorData(input) as *mut f32, len);
    let bias_raw: &mut [f32] = std::slice::from_raw_parts_mut(TF_TensorData(bias) as *mut f32, len);
    let output_raw: &mut [f32] =
        std::slice::from_raw_parts_mut(TF_TensorData(output) as *mut f32, len);

    for i in 0..input_dims.n as usize {
        for j in 0..input_dims.h as usize {
            for k in 0..input_dims.w as usize {
                for (l, current_bias) in bias_raw.iter().enumerate().take(input_dims.c as usize) {
                    let x = offset_from_tensor_coordinates(&dims, &format, i, j, k, l);
                    output_raw[x] = input_raw[x] + current_bias;
                }
            }
        }
    }
}

unsafe extern "C" fn delete(kernel: *mut BiasAddKernel) {
    std::mem::drop(Box::from_raw(kernel))
}
