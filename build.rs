fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let mut build = cc::Build::new();
    build
        .include("include")
        .include("src")
        .file("src/bpe.c")
        .file("src/loader.c")
        .file("src/pair_hashmap.c")
        .file("src/pretokenize.c")
        .file("src/pretokenize_neon.c");

    if target_arch == "x86_64" {
        let mut avx2_build = cc::Build::new();
        avx2_build
            .include("include")
            .include("src")
            .file("src/pretokenize_avx2.c");
        
        if avx2_build.get_compiler().is_like_msvc() {
            avx2_build.flag("/arch:AVX2");
        } else {
            avx2_build.flag("-mavx2");
        }
        avx2_build.compile("pretokenize_avx2");
    }

    build.compile("speed_token_c");

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=include");
}
