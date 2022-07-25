use crate::{bindings::raw::*, DEVICE_NAME, DEVICE_TYPE, EMPTY_CSTR};

use std::{os::raw::c_void, ptr::null_mut};

#[no_mangle]
unsafe extern "C" fn SE_InitPlugin(
    params: *mut SE_PlatformRegistrationParams,
    status: *mut TF_Status,
) {
    (*params).struct_size = std::mem::size_of::<SE_PlatformRegistrationParams>() as u64;
    (*params).destroy_platform = Some(plugin_destroy_platform);
    (*params).destroy_platform_fns = Some(plugin_destroy_platform_fns);

    (*(*params).platform).struct_size = std::mem::size_of::<SP_Platform>() as u64;
    (*(*params).platform).name = DEVICE_NAME.as_ptr() as *const i8;
    (*(*params).platform).type_ = DEVICE_TYPE.as_ptr() as *const i8;

    (*(*params).platform_fns).struct_size = std::mem::size_of::<SP_PlatformFns>() as u64;
    (*(*params).platform_fns).get_device_count = Some(plugin_get_device_count);
    (*(*params).platform_fns).create_device = Some(plugin_create_device);
    (*(*params).platform_fns).destroy_device = Some(plugin_destroy_device);
    (*(*params).platform_fns).create_device_fns = Some(plugin_create_device_fns);
    (*(*params).platform_fns).destroy_device_fns = Some(plugin_destroy_device_fns);
    (*(*params).platform_fns).create_stream_executor = Some(plugin_create_stream_executor);
    (*(*params).platform_fns).destroy_stream_executor = Some(plugin_destroy_stream_executor);
    (*(*params).platform_fns).create_timer_fns = Some(plugin_create_timer_fns);
    (*(*params).platform_fns).destroy_timer_fns = Some(plugin_destroy_timer_fns);

    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

extern "C" fn plugin_destroy_platform(_platform: *mut SP_Platform) {}
extern "C" fn plugin_destroy_platform_fns(_platform_fns: *mut SP_PlatformFns) {}

unsafe extern "C" fn plugin_get_device_count(
    _platform: *const SP_Platform,
    device_count: *mut i32,
    status: *mut TF_Status,
) {
    *device_count = 1;
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_create_device(
    _platform: *const SP_Platform,
    params: *mut SE_CreateDeviceParams,
    status: *mut TF_Status,
) {
    (*(*params).device).struct_size = std::mem::size_of::<SP_Device>() as u64;
    (*(*params).device).device_handle = Box::into_raw(Box::new("magic".to_owned())) as *mut c_void;

    (*(*params).device).ordinal = (*params).ordinal;
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_destroy_device(_platform: *const SP_Platform, device: *mut SP_Device) {
    std::mem::drop(Box::from_raw((*device).device_handle));

    (*device).device_handle = null_mut();
    (*device).ordinal = -1;
}

unsafe extern "C" fn plugin_create_device_fns(
    _platform: *const SP_Platform,
    params: *mut SE_CreateDeviceFnsParams,
    status: *mut TF_Status,
) {
    (*(*params).device_fns).struct_size = std::mem::size_of::<SP_DeviceFns>() as u64;
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

extern "C" fn plugin_destroy_device_fns(
    _platform: *const SP_Platform,
    _device_fns: *mut SP_DeviceFns,
) {
}

unsafe extern "C" fn plugin_create_stream_executor(
    _platform: *const SP_Platform,
    params: *mut SE_CreateStreamExecutorParams,
    status: *mut TF_Status,
) {
    (*(*params).stream_executor).struct_size = std::mem::size_of::<SP_StreamExecutor>() as u64;
    (*(*params).stream_executor).allocate = Some(plugin_allocate);
    (*(*params).stream_executor).deallocate = Some(plugin_deallocate);
    (*(*params).stream_executor).host_memory_allocate = Some(plugin_host_memory_allocate);
    (*(*params).stream_executor).host_memory_deallocate = Some(plugin_host_memory_deallocate);
    (*(*params).stream_executor).get_allocator_stats = Some(plugin_get_allocator_stats);
    (*(*params).stream_executor).device_memory_usage = Some(plugin_device_memory_usage);

    (*(*params).stream_executor).create_stream = Some(plugin_create_stream);
    (*(*params).stream_executor).destroy_stream = Some(plugin_destroy_stream);
    (*(*params).stream_executor).create_stream_dependency = Some(plugin_create_stream_dependency);
    (*(*params).stream_executor).get_stream_status = Some(plugin_get_stream_status);
    (*(*params).stream_executor).create_event = Some(plugin_create_event);
    (*(*params).stream_executor).destroy_event = Some(plugin_destroy_event);
    (*(*params).stream_executor).get_event_status = Some(plugin_get_event_status);
    (*(*params).stream_executor).record_event = Some(plugin_record_event);
    (*(*params).stream_executor).wait_for_event = Some(plugin_wait_for_event);
    (*(*params).stream_executor).create_timer = Some(plugin_create_timer);
    (*(*params).stream_executor).destroy_timer = Some(plugin_destroy_timer);
    (*(*params).stream_executor).start_timer = Some(plugin_start_timer);
    (*(*params).stream_executor).stop_timer = Some(plugin_stop_timer);

    (*(*params).stream_executor).memcpy_dtoh = Some(plugin_memcpy_dtoh);
    (*(*params).stream_executor).memcpy_htod = Some(plugin_memcpy_htod);
    (*(*params).stream_executor).memcpy_dtod = Some(plugin_memcpy_dtod);
    (*(*params).stream_executor).sync_memcpy_dtoh = Some(plugin_sync_memcpy_dtoh);
    (*(*params).stream_executor).sync_memcpy_htod = Some(plugin_sync_memcpy_htod);
    (*(*params).stream_executor).sync_memcpy_dtod = Some(plugin_sync_memcpy_dtod);

    // TODO(plugin): Fill the function for block stream
    (*(*params).stream_executor).block_host_until_done = Some(plugin_block_host_until_done);
    (*(*params).stream_executor).block_host_for_event = Some(plugin_block_host_for_event);

    (*(*params).stream_executor).synchronize_all_activity = Some(plugin_synchronize_all_activity);

    (*(*params).stream_executor).mem_zero = Some(plugin_mem_zero);
    (*(*params).stream_executor).memset = Some(plugin_memset);
    (*(*params).stream_executor).memset32 = Some(plugin_memset32);

    (*(*params).stream_executor).host_callback = Some(plugin_host_callback);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_allocate(
    _device: *const SP_Device,
    size: u64,
    _memory_space: i64,
    mem: *mut SP_DeviceMemoryBase,
) {
    (*mem).struct_size = std::mem::size_of::<SP_DeviceMemoryBase>() as u64;
    (*mem).opaque = libc::malloc(size as libc::size_t) as *mut std::ffi::c_void;
    (*mem).size = size;
}

unsafe extern "C" fn plugin_deallocate(_device: *const SP_Device, mem: *mut SP_DeviceMemoryBase) {
    libc::free((*mem).opaque);
    (*mem).opaque = null_mut();
    (*mem).size = 0;
}

unsafe extern "C" fn plugin_host_memory_allocate(
    _device: *const SP_Device,
    size: u64,
) -> *mut std::ffi::c_void {
    libc::malloc(size as libc::size_t) as *mut std::ffi::c_void
}

unsafe extern "C" fn plugin_host_memory_deallocate(
    _device: *const SP_Device,
    mem: *mut std::ffi::c_void,
) {
    libc::free(mem);
}

unsafe extern "C" fn plugin_get_allocator_stats(
    _device: *const SP_Device,
    stats: *mut SP_AllocatorStats,
) -> u8 {
    (*stats).struct_size = std::mem::size_of::<SP_AllocatorStats>() as u64;
    // FIXME
    (*stats).bytes_in_use = 123;

    1
}

unsafe extern "C" fn plugin_device_memory_usage(
    _device: *const SP_Device,
    free: *mut i64,
    total: *mut i64,
) -> u8 {
    // FIXME
    *free = 256_000_000;
    *total = 512_000_000;
    1
}

unsafe extern "C" fn plugin_create_stream(
    device: *const SP_Device,
    stream: *mut SP_Stream,
    status: *mut TF_Status,
) {
    *stream = Box::into_raw(Box::new(SP_Stream_st {
        stream_handle: (*device).device_handle,
    }));

    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

// Destroys SP_Stream and deallocates any underlying resources.
unsafe extern "C" fn plugin_destroy_stream(_device: *const SP_Device, stream: SP_Stream) {
    std::mem::drop(Box::from_raw(stream))
}

unsafe extern "C" fn plugin_create_stream_dependency(
    _device: *const SP_Device,
    _dependent: SP_Stream,
    _other: SP_Stream,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

// Without blocking the device, retrieve the current stream status.
unsafe extern "C" fn plugin_get_stream_status(
    _device: *const SP_Device,
    _stream: SP_Stream,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_create_event(
    _device: *const SP_Device,
    _event: *mut SP_Event,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

// Destroy SE_Event and perform any platform-specific deallocation and
// cleanup of an event.
extern "C" fn plugin_destroy_event(_device: *const SP_Device, _event: SP_Event) {}

// Requests the current status of the event from the underlying platform.
extern "C" fn plugin_get_event_status(
    _device: *const SP_Device,
    _event: SP_Event,
) -> SE_EventStatus {
    SE_EVENT_COMPLETE
}

// Inserts the specified event at the end of the specified stream.
unsafe extern "C" fn plugin_record_event(
    _device: *const SP_Device,
    _stream: SP_Stream,
    _event: SP_Event,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

// Wait for the specified event at the end of the specified stream.
unsafe extern "C" fn plugin_wait_for_event(
    _device: *const SP_Device,
    _stream: SP_Stream,
    _event: SP_Event,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

extern "C" fn plugin_destroy_stream_executor(
    _platform: *const SP_Platform,
    _stream_executor: *mut SP_StreamExecutor,
) {
}

unsafe extern "C" fn plugin_create_timer(
    _device: *const SP_Device,
    _timer: *mut SP_Timer,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

// Destroy timer and deallocates timer resources on the underlying platform.
extern "C" fn plugin_destroy_timer(_device: *const SP_Device, _timer: SP_Timer) {}

// Records a start event for an interval timer.
unsafe extern "C" fn plugin_start_timer(
    _device: *const SP_Device,
    _stream: SP_Stream,
    _timer: SP_Timer,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

// Records a stop event for an interval timer.
unsafe extern "C" fn plugin_stop_timer(
    _device: *const SP_Device,
    _stream: SP_Stream,
    _timer: SP_Timer,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_memcpy_dtoh(
    _device: *const SP_Device,
    _stream: SP_Stream,
    host_dst: *mut std::ffi::c_void,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    status: *mut TF_Status,
) {
    libc::memcpy(host_dst, (*device_src).opaque, size as usize);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}
unsafe extern "C" fn plugin_sync_memcpy_dtoh(
    _device: *const SP_Device,
    host_dst: *mut std::ffi::c_void,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    status: *mut TF_Status,
) {
    libc::memcpy(host_dst, (*device_src).opaque, size as usize);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_memcpy_dtod(
    _device: *const SP_Device,
    _stream: SP_Stream,
    device_dst: *mut SP_DeviceMemoryBase,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    status: *mut TF_Status,
) {
    libc::memcpy((*device_dst).opaque, (*device_src).opaque, size as usize);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}
unsafe extern "C" fn plugin_sync_memcpy_dtod(
    _device: *const SP_Device,
    device_dst: *mut SP_DeviceMemoryBase,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    status: *mut TF_Status,
) {
    libc::memcpy((*device_dst).opaque, (*device_src).opaque, size as usize);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_memcpy_htod(
    _device: *const SP_Device,
    _stream: SP_Stream,
    device_dst: *mut SP_DeviceMemoryBase,
    host_src: *const std::ffi::c_void,
    size: u64,
    status: *mut TF_Status,
) {
    libc::memcpy((*device_dst).opaque, host_src, size as usize);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}
unsafe extern "C" fn plugin_sync_memcpy_htod(
    _device: *const SP_Device,
    device_dst: *mut SP_DeviceMemoryBase,
    host_src: *const std::ffi::c_void,
    size: u64,
    status: *mut TF_Status,
) {
    libc::memcpy((*device_dst).opaque, host_src, size as usize);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_block_host_for_event(
    _device: *const SP_Device,
    _event: SP_Event,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_block_host_until_done(
    _device: *const SP_Device,
    _stream: SP_Stream,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

unsafe extern "C" fn plugin_synchronize_all_activity(
    _device: *const SP_Device,
    status: *mut TF_Status,
) {
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}
unsafe extern "C" fn plugin_mem_zero(
    _device: *const SP_Device,
    _stream: SP_Stream,
    location: *mut SP_DeviceMemoryBase,
    size: u64,
    status: *mut TF_Status,
) {
    plugin_memset32(_device, _stream, location, 0, size, status);
}

unsafe extern "C" fn plugin_memset(
    _device: *const SP_Device,
    _stream: SP_Stream,
    location: *mut SP_DeviceMemoryBase,
    pattern: u8,
    size: u64,
    status: *mut TF_Status,
) {
    let pattern: u32 = pattern.into();
    plugin_memset32(
        _device,
        _stream,
        location,
        pattern << 24 | pattern << 16 | pattern << 8 | pattern,
        size,
        status,
    );
}

unsafe extern "C" fn plugin_memset32(
    _device: *const SP_Device,
    _stream: SP_Stream,
    location: *mut SP_DeviceMemoryBase,
    pattern: u32,
    size: u64,
    status: *mut TF_Status,
) {
    libc::memset((*location).opaque, pattern as i32, size as usize);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

extern "C" fn plugin_host_callback(
    _device: *const SP_Device,
    _stream: SP_Stream,
    _callback_fn: SE_StatusCallbackFn,
    _callback_arg: *mut std::ffi::c_void,
) -> u8 {
    1
}

extern "C" fn nanoseconds(timer: *mut SP_Timer_st) -> u64 {
    unsafe { (*timer).timer_handle as u64 }
}

unsafe extern "C" fn plugin_create_timer_fns(
    _platform: *const SP_Platform,
    timer_fns: *mut SP_TimerFns,
    status: *mut TF_Status,
) {
    (*timer_fns).nanoseconds = Some(nanoseconds);
    TF_SetStatus(status, TF_OK, EMPTY_CSTR.as_ptr() as *const i8);
}

extern "C" fn plugin_destroy_timer_fns(
    _platform: *const SP_Platform,
    _timer_fns: *mut SP_TimerFns,
) {
}
