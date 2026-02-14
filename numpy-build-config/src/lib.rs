use self::impl_::NumpyVersion;

mod impl_;

pub fn use_numpy_cfgs() {
    print_expected_features();
    print_enabled_features();
}

fn print_expected_features() {
    for version in NumpyVersion::supported() {
        println!(
            "cargo:rustc-check-cfg=cfg(Numpy_{}_{})",
            version.major, version.minor
        );
    }
}

fn print_enabled_features() {
    for version in NumpyVersion::enabled() {
        println!("cargo:rustc-cfg=Numpy_{}_{}", version.major, version.minor);
    }
}
