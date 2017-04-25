
extern crate cpython;

use std::env::var;
use std::path::PathBuf;
use std::process::Command;
use cpython::*;

fn get_module_path(py: Python) -> PyResult<String> {
    let ma = py.import("numpy.core.multiarray")?;
    let path: String = ma.get(py, "__file__")?.extract(py)?;
    Ok(path)
}

fn main() {
    let output = PathBuf::from(var("OUT_DIR").unwrap().replace(r"\", "/"));
    let gil = Python::acquire_gil();
    let path = PathBuf::from(get_module_path(gil.python()).unwrap());
    let dir = path.parent().unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();

    Command::new("ln")
        .arg("-s")
        .arg(path.to_str().unwrap())
        .arg(output.join("libmultiarray.so").to_str().unwrap())
        .output()
        .expect("Faild to create symbolic link");
    println!("cargo:rustc-link-search={}", output.display());
    println!("cargo:rustc-link-lib=dylib=multiarray");
}
