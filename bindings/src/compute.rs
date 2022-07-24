use super::raw::*;

impl TF_Status {
    pub fn is_ok(self: *mut Self) -> bool {
        unsafe { TF_GetCode(self) == TF_OK }
    }
}

impl TF_OpKernelConstruction {
    pub fn get_attr_size(self: *mut Self, attr_name: &str) -> Result<(i32, i32), *mut TF_Status> {
        let status_ptr = unsafe { TF_NewStatus() };
        let mut list_size = 0i32;
        let mut total_size = 0i32;

        unsafe {
            TF_OpKernelConstruction_GetAttrSize(
                self,
                attr_name.as_ptr() as *const i8,
                &mut list_size,
                &mut total_size,
                status_ptr,
            );
        }

        if status_ptr.is_ok() {
            Ok((list_size, total_size))
        } else {
            Err(status_ptr)
        }
    }

    pub fn get_attr_string(
        self: *mut Self,
        attr_name: &'static str,
    ) -> Result<String, *mut TF_Status> {
        assert!(attr_name.ends_with("\0"), "Strings must be zero-terminated");

        let (list_size, total_size) = match self.get_attr_size(attr_name) {
            Ok(value) => value,
            Err(value) => return Err(value),
        };

        assert!(list_size == -1);

        let status_ptr = unsafe { TF_NewStatus() };
        let mut val = Vec::with_capacity(total_size as usize);

        unsafe {
            TF_OpKernelConstruction_GetAttrString(
                self,
                attr_name.as_ptr() as *const i8,
                val.as_mut_ptr() as *mut i8,
                total_size as u64,
                status_ptr,
            );
        }
        if status_ptr.is_ok() {
            unsafe {
                val.set_len(total_size as usize);
            }
            // Safety: will panic on invalid string
            Ok(String::from_utf8(val).unwrap())
        } else {
            Err(status_ptr)
        }
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
        dims: &Vec<i64>,
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

    pub fn dims_named(self: *mut Self, format: &String) -> NamedDims {
        let dims = self.dims();

        assert!(dims.len() == 4);
        assert!(format.len() == 4);

        let (ni, hi, wi, ci) = get_format_indices(format);

        NamedDims {
            n: dims.get(ni).unwrap().clone(),
            ni,
            h: dims.get(hi).unwrap().clone(),
            hi,
            w: dims.get(wi).unwrap().clone(),
            wi,
            c: dims.get(ci).unwrap().clone(),
            ci,
        }
    }
}

// Computes raw offset according to format
// TODO: unit test
pub fn index_from_nhwc_coordinates(
    dims: &Vec<i64>,
    format: &String,
    n: usize,
    h: usize,
    w: usize,
    c: usize,
) -> usize {
    assert!(dims.len() == 4);
    assert!(format.len() == 4);

    let dims: Box<[i64]> = dims.clone().into_boxed_slice();
    let (ni, hi, wi, ci) = get_format_indices(format);

    axis_offset(n, ni, dims.clone())
        + axis_offset(h, hi, dims.clone())
        + axis_offset(w, wi, dims.clone())
        + axis_offset(c, ci, dims.clone())
}

fn axis_offset(a: usize, index: usize, dims: Box<[i64]>) -> usize {
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

fn get_format_indices(format: &String) -> (usize, usize, usize, usize) {
    (
        format.find("N").expect("N should be in the format"),
        format.find("H").expect("H should be in the format"),
        format.find("W").expect("W should be in the format"),
        format.find("C").expect("C should be in the format"),
    )
}
