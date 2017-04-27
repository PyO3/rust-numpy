
extern crate glob;

use std::fs::canonicalize;
use std::path::PathBuf;
use std::process::Command;
use glob::glob;


fn search_lib(libpath: &PathBuf, regex: &str) -> PathBuf {
    glob(libpath.join(regex).to_str().unwrap()).unwrap().last().unwrap().unwrap()
}

fn main() {
    let source = PathBuf::from("numpy");
    let libpath = canonicalize(PathBuf::from("numpy/build/lib.linux-x86_64-3.6/numpy/core")).unwrap();

    Command::new("python").args(&["setup.py", "build"]).current_dir(&source).status().expect("Failed to build numpy");
    let multiarray = search_lib(&libpath, "multiarray.*.so");
    let umath = search_lib(&libpath, "umath.*.so");

    Command::new("ln")
        .arg("-sf")
        .arg(multiarray.file_name().unwrap())
        .arg("libmultiarray.so")
        .current_dir(&libpath)
        .status()
        .expect("Failed to create symlink");
    Command::new("ln")
        .arg("-sf")
        .arg(umath.file_name().unwrap())
        .arg("libumath.so")
        .current_dir(&libpath)
        .status()
        .expect("Failed to create symlink");
    println!("cargo:rustc-link-search={}", libpath.display());
    println!("cargo:rustc-link-lib=dylib=multiarray");
    println!("cargo:rustc-link-lib=dylib=umath");
}
