use ::std::os::raw::c_void;
use std::collections::HashMap;

use super::raw::*;

pub struct KernelBuilder {
    kernel_name: &'static str,
    op_name: &'static str,
    device_type: &'static str,
    constraints: HashMap<&'static str, TF_DataType>,
    create_fn: Option<unsafe extern "C" fn(*mut TF_OpKernelConstruction) -> *mut c_void>,
    compute_fn: Option<unsafe extern "C" fn(*mut c_void, *mut TF_OpKernelContext)>,
    delete_fn: Option<unsafe extern "C" fn(*mut c_void)>,
}

impl KernelBuilder {
    pub fn new(
        kernel_name: &'static str,
        op_name: &'static str,
        device_type: &'static str,
    ) -> Self {
        assert!(
            kernel_name.ends_with("\0"),
            "Strings must be zero-terminated"
        );
        assert!(op_name.ends_with("\0"), "Strings must be zero-terminated");
        assert!(
            device_type.ends_with("\0"),
            "Strings must be zero-terminated"
        );
        Self {
            kernel_name,
            op_name,
            device_type,
            constraints: HashMap::new(),
            create_fn: None,
            compute_fn: None,
            delete_fn: None,
        }
    }

    pub fn constraint(mut self, name: &'static str, dt: TF_DataType) -> Self {
        assert!(name.ends_with("\0"), "Strings must be zero-terminated");
        self.constraints.insert(name, dt);
        self
    }

    pub fn create(
        mut self,
        function: unsafe extern "C" fn(*mut TF_OpKernelConstruction) -> *mut c_void,
    ) -> Self {
        self.create_fn = Some(function);
        self
    }

    pub fn compute(
        mut self,
        function: unsafe extern "C" fn(*mut c_void, *mut TF_OpKernelContext),
    ) -> Self {
        self.compute_fn = Some(function);
        self
    }

    pub fn delete(mut self, function: unsafe extern "C" fn(*mut c_void)) -> Self {
        self.delete_fn = Some(function);
        self
    }

    pub fn register(self) {
        unsafe {
            let builder = TF_NewKernelBuilder(
                self.kernel_name.as_ptr() as *const i8,
                self.device_type.as_ptr() as *const i8,
                self.create_fn,
                self.compute_fn,
                self.delete_fn,
            );

            for (name, dt) in self.constraints {
                let status = TF_NewStatus();
                TF_KernelBuilder_TypeConstraint(builder, name.as_ptr() as *const i8, dt, status);
                if TF_OK != TF_GetCode(status) {
                    eprintln!(
                        "Error while registering {} kernel with attribute {}",
                        self.op_name, name
                    );
                    return;
                }
            }

            let status = TF_NewStatus();
            TF_RegisterKernelBuilder(self.op_name.as_ptr() as *const i8, builder, status);
            if TF_OK != TF_GetCode(status) {
                eprintln!("Error while registering {} kernel", self.op_name);
                return;
            }
        }
    }
}
