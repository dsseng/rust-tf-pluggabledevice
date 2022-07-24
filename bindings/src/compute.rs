use super::raw::*;

impl TF_Status {
    pub fn is_ok(self: *mut Self) -> bool {
        unsafe { TF_GetCode(self) == TF_OK }
    }
}

impl TF_OpKernelContext {
    // Still unsafe because of type transmutation
    pub unsafe fn get_stream<T>(self: *mut Self) -> Result<*mut T, *mut TF_Status> {
        let status_ptr = TF_NewStatus();
        let stream = TF_GetStream(self, status_ptr);

        if status_ptr.is_ok() {
            Ok((*stream).stream_handle as *mut T)
        } else {
            Err(status_ptr)
        }
    }

    pub fn get_input(self: *mut Self, i: i32) -> Result<*mut TF_Tensor, *mut TF_Status> {
        unsafe {
            let mut input_ptr = 0 as *mut TF_Tensor;
            let status_ptr = TF_NewStatus();
            TF_GetInput(self, i, &mut input_ptr, status_ptr);

            if status_ptr.is_ok() {
                Ok(input_ptr)
            } else {
                Err(status_ptr)
            }
        }
    }

    pub fn failure(self: *mut Self, status: *mut TF_Status) {
        unsafe { TF_OpKernelContext_Failure(self, status) }
    }

    pub fn allocate_output(
        self: *mut Self,
        i: i32,
        dims: Vec<i64>,
        len: u64,
    ) -> Result<*mut TF_Tensor, *mut TF_Status> {
        unsafe {
            let status_ptr = TF_NewStatus();
            let output = TF_AllocateOutput(
                self,
                i,
                TF_ExpectedOutputDataType(self, i),
                dims.as_ptr(),
                dims.len() as i32,
                len,
                status_ptr,
            );

            if status_ptr.is_ok() {
                Ok(output)
            } else {
                Err(status_ptr)
            }
        }
    }
}

impl TF_Tensor {
    pub fn element_count(self: *mut Self) -> i64 {
        unsafe { return TF_TensorElementCount(self) }
    }

    pub fn num_dims(self: *mut Self) -> i32 {
        unsafe { return TF_NumDims(self) }
    }

    pub fn dims(self: *mut Self) -> Vec<i64> {
        let ndim = self.num_dims();
        let mut dims: Vec<i64> = Vec::new();

        unsafe {
            for i in 0..ndim {
                dims.push(TF_Dim(self, i));
            }
        }

        dims
    }
}
