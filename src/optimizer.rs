// FIXME: double free or corruption (!prev)
// needs a proper implementation of optimizer, but I didn't port that
// also need to decode protobuf, process it and then encode back to protobuf

pub mod optimizer {
    // use crate::{bindings::raw::*, DEVICE_TYPE, EMPTY_CSTR};
    // use std::ptr::null_mut;

    // #[no_mangle]
    // pub unsafe extern "C" fn TF_InitGraph(
    //     params: *mut TP_OptimizerRegistrationParams,
    //     status: *mut TF_Status,
    // ) {
    //     (*params).struct_size = std::mem::size_of::<TP_OptimizerRegistrationParams>() as u64;
    //     (*(*params).optimizer).struct_size = std::mem::size_of::<TP_Optimizer>() as u64;
    //     (*(*params).optimizer_configs).struct_size =
    //         std::mem::size_of::<TP_OptimizerConfigs>() as u64;

    //     (*(*params).optimizer_configs).remapping = TF_TriState_Off;
    //     (*(*params).optimizer_configs).layout_optimizer = TF_TriState_Off;
    //     (*params).device_type = DEVICE_TYPE.as_ptr() as *const i8;

    //     (*(*params).optimizer).create_func = Some(plugin_create_func);
    //     (*(*params).optimizer).optimize_func = Some(plugin_optimize_func);
    //     (*(*params).optimizer).destroy_func = Some(plugin_destroy_func);
    //     TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
    // }

    // extern "C" fn plugin_create_func() -> *mut ::std::os::raw::c_void {
    //     null_mut()
    // }
    // unsafe extern "C" fn plugin_optimize_func(
    //     _arg1: *mut ::std::os::raw::c_void,
    //     arg2: *const TF_Buffer,
    //     _arg3: *const TF_GrapplerItem,
    //     arg4: *mut TF_Buffer,
    //     status: *mut TF_Status,
    // ) {
    //     *arg4 = *arg2;
    //     TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
    // }
    // extern "C" fn plugin_destroy_func(_arg1: *mut ::std::os::raw::c_void) {}
}
