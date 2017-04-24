
extern crate gcc;

fn main() {
    gcc::Config::new()
        .file("src/array_api.c")
        .flag("-fkeep-inline-functions")
        .include("/usr/include/python3.6m")
        .include("/usr/lib/python3.6/site-packages/numpy/core/include")
        .compile("libnumpy_c_api.a");
}
