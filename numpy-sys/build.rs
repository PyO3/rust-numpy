
extern crate glob;

use std::fs::canonicalize;
use std::path::PathBuf;
use std::process::Command;
use glob::glob;


fn main() {
    let source = PathBuf::from("numpy");
    let libpath = canonicalize(PathBuf::from("numpy/build/lib.linux-x86_64-3.6/numpy/core")).unwrap();

    Command::new("python").args(&["setup.py", "build"]).current_dir(&source).status().expect("Failed to build numpy");

    let lib: PathBuf = glob(libpath.join("multiarray.*.so").to_str().unwrap()).unwrap().last().unwrap().unwrap();
    let name = lib.file_name().unwrap();

    Command::new("ln")
        .arg("-sf")
        .arg(name)
        .arg("libmultiarray.so")
        .current_dir(&libpath)
        .status()
        .expect("Failed to create symlink");
    println!("cargo:rustc-link-search={}", libpath.display());
    println!("cargo:rustc-link-lib=dylib=multiarray");
}
