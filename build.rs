use std::process::{Command, Stdio};
use std::str;
fn is_python2() -> bool {
    let script = r"
import sys
print(sys.version_info[0])
";
    let out = Command::new("python")
        .args(&["-c", script])
        .stderr(Stdio::inherit())
        .output()
        .expect("Failed to run the python interpreter");
    let version = str::from_utf8(&out.stdout).unwrap();
    version.starts_with("2")
}

fn cfg(python2: bool) {
    if python2 {
        println!("cargo:rustc-cfg=Py_2");
    } else {
        println!("cargo:rustc-cfg=Py_3");
    }
}

fn main() {
    if cfg!(feature = "python3") {
        cfg(false);
    } else if cfg!(feature = "python2") {
        cfg(true);
    } else {
        cfg(is_python2());
    }
}
