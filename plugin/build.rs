use std::path::PathBuf;

fn main() {
    // TODO: support other versions aside from 3.10
    println!(
        "cargo:rustc-link-search={}",
        PathBuf::from("./venv/lib/python3.10/site-packages/tensorflow/")
            .to_str()
            .unwrap()
    );
    // TODO: support other OSs, not only Linux
    println!("cargo:rustc-link-arg=-l:libtensorflow_framework.so.2");
}
