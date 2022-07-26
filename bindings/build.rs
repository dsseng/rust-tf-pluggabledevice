use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TF_INCLUDE_PATH");
    println!("cargo:rerun-if-env-changed=TF_LIBRARY_PATH");

    // TODO: support other versions aside from 3.10
    println!(
        "cargo:rustc-link-search={}",
        PathBuf::from(
            env::var("TF_LIBRARY_PATH")
                .unwrap_or_else(|_| "./venv/lib/python3.10/site-packages/tensorflow/".to_owned())
        )
        .to_str()
        .unwrap()
    );
    // TODO: support other OSs, not only Linux
    println!("cargo:rustc-cdylib-link-arg=-l:libtensorflow_framework.so.2");

    println!("cargo:rerun-if-changed=wrapper.hh");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.

    // TODO: support other versions aside from 3.10
    let bindings = bindgen::Builder::default()
        .clang_arg(
            "-I".to_owned()
                + PathBuf::from(env::var("TF_INCLUDE_PATH").unwrap_or_else(|_| {
                    "../venv/lib/python3.10/site-packages/tensorflow/include/".to_owned()
                }))
                .to_str()
                .unwrap(),
        )
        // Exported by plugin, not imported
        .blocklist_function("SE_InitPlugin")
        .blocklist_function("TF_InitGraph")
        .blocklist_function("TF_InitKernel")
        // TF_Code_TF_OK -> TF_OK, better matches C
        .prepend_enum_name(false)
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.hh")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings to TensorFlow C API");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
