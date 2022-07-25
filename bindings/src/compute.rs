use super::raw::*;

impl TF_Status {
    /// # Safety
    ///
    /// Should be called on a valid, initialized TF_Status
    pub unsafe fn is_ok(self: *mut Self) -> bool {
        TF_GetCode(self) == TF_OK
    }
}

impl TF_OpKernelConstruction {
    /// # Safety
    ///
    /// Should be called on a TF_OpKernelConstruction received by kernel creation function
    pub unsafe fn get_attr_size(
        self: *mut Self,
        attr_name: &str,
    ) -> Result<(i32, i32), *mut TF_Status> {
        let status = TF_NewStatus();
        let mut list_size = 0i32;
        let mut total_size = 0i32;

        TF_OpKernelConstruction_GetAttrSize(
            self,
            attr_name.as_ptr() as *const i8,
            &mut list_size,
            &mut total_size,
            status,
        );

        if status.is_ok() {
            TF_DeleteStatus(status);
            Ok((list_size, total_size))
        } else {
            Err(status)
        }
    }

    /// # Safety
    ///
    /// Should be called on a TF_OpKernelConstruction received by kernel creation function
    pub unsafe fn get_attr_string(
        self: *mut Self,
        attr_name: &'static str,
    ) -> Result<String, *mut TF_Status> {
        assert!(attr_name.ends_with('\0'), "Strings must be zero-terminated");

        let (list_size, total_size) = match self.get_attr_size(attr_name) {
            Ok(value) => value,
            Err(value) => return Err(value),
        };

        assert!(list_size == -1);

        let status = TF_NewStatus();
        let mut val = Vec::with_capacity(total_size as usize);

        TF_OpKernelConstruction_GetAttrString(
            self,
            attr_name.as_ptr() as *const i8,
            val.as_mut_ptr() as *mut i8,
            total_size as u64,
            status,
        );

        if status.is_ok() {
            val.set_len(total_size as usize);
            TF_DeleteStatus(status);

            // Safety: will panic on invalid string
            Ok(String::from_utf8(val).unwrap())
        } else {
            Err(status)
        }
    }
}

impl TF_OpKernelContext {
    /// # Safety
    ///
    /// Should be called on a TF_OpKernelContext received by kernel compute function
    /// Type of stream you pass must be the same as one you used in your plugin's stream creation function
    pub unsafe fn get_stream<T>(self: *mut Self) -> Result<*mut T, *mut TF_Status> {
        let status = TF_NewStatus();
        let stream = TF_GetStream(self, status);

        if status.is_ok() {
            TF_DeleteStatus(status);
            Ok((*stream).stream_handle as *mut T)
        } else {
            Err(status)
        }
    }

    /// # Safety
    ///
    /// Should be called on a TF_OpKernelContext received by kernel compute function
    pub unsafe fn get_input(self: *mut Self, i: i32) -> Result<*mut TF_Tensor, *mut TF_Status> {
        let mut input_ptr = std::ptr::null_mut::<TF_Tensor>();
        let status = TF_NewStatus();
        TF_GetInput(self, i, &mut input_ptr, status);

        if status.is_ok() {
            TF_DeleteStatus(status);
            Ok(input_ptr)
        } else {
            Err(status)
        }
    }

    /// # Safety
    ///
    /// Should be called on a TF_OpKernelContext received by kernel compute function
    /// Should receive a valid TF_Status
    pub unsafe fn failure(self: *mut Self, status: *mut TF_Status) {
        TF_OpKernelContext_Failure(self, status)
    }

    /// # Safety
    ///
    /// Should be called on a TF_OpKernelContext received by kernel compute function
    pub unsafe fn allocate_output(
        self: *mut Self,
        i: i32,
        dims: &Vec<i64>,
        len: u64,
    ) -> Result<*mut TF_Tensor, *mut TF_Status> {
        let status = TF_NewStatus();
        let output = TF_AllocateOutput(
            self,
            i,
            TF_ExpectedOutputDataType(self, i),
            dims.as_ptr(),
            dims.len() as i32,
            len,
            status,
        );

        if status.is_ok() {
            TF_DeleteStatus(status);
            Ok(output)
        } else {
            Err(status)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NamedDims {
    pub n: i64,
    pub h: i64,
    pub w: i64,
    pub c: i64,
    pub ni: usize,
    pub hi: usize,
    pub wi: usize,
    pub ci: usize,
}

impl TF_Tensor {
    /// # Safety
    ///
    /// Should be called on an initialized TF_Tensor
    pub unsafe fn element_count(self: *mut Self) -> i64 {
        TF_TensorElementCount(self)
    }

    /// # Safety
    ///
    /// Should be called on an initialized TF_Tensor
    pub unsafe fn num_dims(self: *mut Self) -> i32 {
        TF_NumDims(self)
    }

    /// # Safety
    ///
    /// Should be called on an initialized TF_Tensor
    pub unsafe fn dims(self: *mut Self) -> Vec<i64> {
        let ndim = self.num_dims();
        let mut dims: Vec<i64> = Vec::new();

        for i in 0..ndim {
            dims.push(TF_Dim(self, i));
        }

        dims
    }

    /// # Safety
    ///
    /// Should be called on an initialized TF_Tensor
    pub unsafe fn dims_named(self: *mut Self, format: &String) -> NamedDims {
        let dims = self.dims();

        assert!(dims.len() == 4);
        assert!(format.len() == 4);

        let (ni, hi, wi, ci) = get_format_indices(format);

        NamedDims {
            n: *dims.get(ni).unwrap(),
            ni,
            h: *dims.get(hi).unwrap(),
            hi,
            w: *dims.get(wi).unwrap(),
            wi,
            c: *dims.get(ci).unwrap(),
            ci,
        }
    }
}

// Computes raw offset according to format
pub fn offset_from_tensor_coordinates(
    dims: &[i64],
    format: &str,
    n: usize,
    h: usize,
    w: usize,
    c: usize,
) -> usize {
    assert!(dims.len() == 4);
    assert!(format.len() == 4);
    let (ni, hi, wi, ci) = get_format_indices(format);

    axis_offset(n, ni, dims)
        + axis_offset(h, hi, dims)
        + axis_offset(w, wi, dims)
        + axis_offset(c, ci, dims)
}

fn axis_offset(a: usize, index: usize, dims: &[i64]) -> usize {
    // Next I use leter C as an example
    a * match index {
        // ...C, increment to C gives us 1
        3 => 1,
        // ..C., 1 of C = dims[3] of elements
        2 => dims[3] as usize,
        // .C.., 1 of C = dims[2] of arrays sized dims[3] of elements
        1 => (dims[2] * dims[3]) as usize,
        // C..., 1 of C = dims[1] of 2D arrays of dims[2] of arrays sized dims[3] of elements
        0 => (dims[1] * dims[2] * dims[3]) as usize,
        _ => unreachable!(),
    }
}

fn get_format_indices(format: &str) -> (usize, usize, usize, usize) {
    (
        format.find('N').expect("N should be in the format"),
        format.find('H').expect("H should be in the format"),
        format.find('W').expect("W should be in the format"),
        format.find('C').expect("C should be in the format"),
    )
}

#[cfg(test)]
mod tests {
    use super::offset_from_tensor_coordinates;

    // Our test tensor is 2x2x2x2
    // n=0 c=0
    // 0 1
    // 2 3
    // n=0 c=1
    // 4 5
    // 6 7
    // n=1 c=0
    // 8 9
    // 10 11
    // n=1 c=1
    // 12 13
    // 14 15

    fn offset_test_base(raw: [u64; 16], dims: [i64; 4], format: &str) {
        let f = offset_from_tensor_coordinates;

        // n=0 c=0
        // 0 1
        // 2 3

        assert_eq!(raw[f(&dims, format, 0, 0, 0, 0)], 0);
        // 1 right
        assert_eq!(raw[f(&dims, format, 0, 0, 1, 0)], 1);
        // 1 down
        assert_eq!(raw[f(&dims, format, 0, 1, 0, 0)], 2);
        // 1 right 1 down
        assert_eq!(raw[f(&dims, format, 0, 1, 1, 0)], 3);

        // n=0 c=1
        // 4 5
        // 6 7

        assert_eq!(raw[f(&dims, format, 0, 0, 0, 1)], 4);
        // 1 right
        assert_eq!(raw[f(&dims, format, 0, 0, 1, 1)], 5);
        // 1 down
        assert_eq!(raw[f(&dims, format, 0, 1, 0, 1)], 6);
        // 1 right 1 down
        assert_eq!(raw[f(&dims, format, 0, 1, 1, 1)], 7);

        // n=1 c=0
        // 8 9
        // 10 11

        assert_eq!(raw[f(&dims, format, 1, 0, 0, 0)], 8);
        // 1 right
        assert_eq!(raw[f(&dims, format, 1, 0, 1, 0)], 9);
        // 1 down
        assert_eq!(raw[f(&dims, format, 1, 1, 0, 0)], 10);
        // 1 right 1 down
        assert_eq!(raw[f(&dims, format, 1, 1, 1, 0)], 11);

        // n=1 c=1
        // 12 13
        // 14 15

        assert_eq!(raw[f(&dims, format, 1, 0, 0, 1)], 12);
        // 1 right
        assert_eq!(raw[f(&dims, format, 1, 0, 1, 1)], 13);
        // 1 down
        assert_eq!(raw[f(&dims, format, 1, 1, 0, 1)], 14);
        // 1 right 1 down
        assert_eq!(raw[f(&dims, format, 1, 1, 1, 1)], 15);
    }

    #[test]
    fn tensor_offset_nchw() {
        // Here is raw data
        // it's first split into 2 halves addressed by n,
        // then 2 by c (select channel), then 2 by h and finally get subpixel by w
        let raw: [u64; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        offset_test_base(raw, [2, 2, 2, 2], "NCHW");
    }

    #[test]
    fn tensor_offset_nhwc() {
        // Here is raw data
        // it's first split into 2 halves addressed by n,
        // then 2 by h, then 2 by w and finally each pixel has 2 channels
        let raw: [u64; 16] = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15];

        offset_test_base(raw, [2, 2, 2, 2], "NHWC");
    }

    #[test]
    fn tensor_offset_chwn() {
        // Here is raw data
        // it's first split into 2 halves addressed by c,
        // then 2 by h, then 2 by w and finally select pixel by n
        let raw: [u64; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

        offset_test_base(raw, [2, 2, 2, 2], "CHWN");
    }

    #[test]
    fn tensor_offset_cnhw() {
        // Here is raw data
        // it's first split into 2 halves addressed by c,
        // then 2 by n, then 2 by h and finally get pixel by w
        let raw: [u64; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15];

        offset_test_base(raw, [2, 2, 2, 2], "CNHW");
    }
}
