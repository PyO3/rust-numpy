#[derive(Debug, Clone, Copy)]
pub struct NumpyVersion {
    pub minor: u32,
    pub major: u32,
}

#[allow(non_snake_case)]
impl NumpyVersion {
    const fn V1(minor: u32) -> Self {
        Self { major: 1, minor }
    }
    const fn V2(minor: u32) -> Self {
        Self { major: 2, minor }
    }
}

impl NumpyVersion {
    /// An iterator over supported versions of numpy API.
    pub fn supported() -> impl Iterator<Item = Self> {
        SUPPORTED_VERSIONS.iter().copied()
    }

    /// An iterator over enabled versions of numpy API.
    pub fn enabled() -> impl Iterator<Item = Self> {
        ENABLED_VERSIONS.iter().copied()
    }
}

const SUPPORTED_VERSIONS: &[NumpyVersion] = &[
    NumpyVersion::V1(15),
    NumpyVersion::V1(16),
    NumpyVersion::V1(17),
    NumpyVersion::V1(18),
    NumpyVersion::V1(19),
    NumpyVersion::V1(20),
    NumpyVersion::V1(21),
    NumpyVersion::V1(22),
    NumpyVersion::V1(23),
    NumpyVersion::V1(24),
    NumpyVersion::V1(25),
    NumpyVersion::V2(0),
    NumpyVersion::V2(1),
    NumpyVersion::V2(2),
    NumpyVersion::V2(3),
    NumpyVersion::V2(4),
];

const ENABLED_VERSIONS: &[NumpyVersion] = &[
    #[cfg(feature = "target-npy115")]
    NumpyVersion::V1(15), // 0x0000000c
    #[cfg(any(
        feature = "target-npy116",
        feature = "target-npy117",
        feature = "target-npy118",
        feature = "target-npy119"
    ))]
    NumpyVersion::V1(16), // 0x0000000d
    #[cfg(any(
        feature = "target-npy116",
        feature = "target-npy117",
        feature = "target-npy118",
        feature = "target-npy119"
    ))]
    NumpyVersion::V1(17), // 0x0000000d
    #[cfg(any(
        feature = "target-npy116",
        feature = "target-npy117",
        feature = "target-npy118",
        feature = "target-npy119"
    ))]
    NumpyVersion::V1(18), // 0x0000000d
    #[cfg(any(
        feature = "target-npy116",
        feature = "target-npy117",
        feature = "target-npy118",
        feature = "target-npy119"
    ))]
    NumpyVersion::V1(19), // 0x0000000d
    #[cfg(any(feature = "target-npy120", feature = "target-npy121"))]
    NumpyVersion::V1(20), // 0x0000000e
    #[cfg(any(feature = "target-npy120", feature = "target-npy121"))]
    NumpyVersion::V1(21), // 0x0000000e
    #[cfg(feature = "target-npy122")]
    NumpyVersion::V1(22), // 0x0000000f
    #[cfg(any(feature = "target-npy123", feature = "target-npy124"))]
    NumpyVersion::V1(23), // 0x00000010
    #[cfg(any(feature = "target-npy123", feature = "target-npy124"))]
    NumpyVersion::V1(24), // 0x00000010
    #[cfg(feature = "target-npy125")]
    NumpyVersion::V1(25), // 0x00000011
    #[cfg(feature = "target-npy20")]
    NumpyVersion::V2(0), // 0x00000012
    #[cfg(any(feature = "target-npy21", feature = "target-npy22"))]
    NumpyVersion::V2(1), // 0x00000013
    #[cfg(any(feature = "target-npy21", feature = "target-npy22"))]
    NumpyVersion::V2(2), // 0x00000013
    #[cfg(feature = "target-npy23")]
    NumpyVersion::V2(3), // 0x00000014
    #[cfg(feature = "target-npy24")]
    NumpyVersion::V2(4), // 0x00000015
];
