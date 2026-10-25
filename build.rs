fn main() {
    println!("cargo:rerun-if-changed=native/metal_bridge.mm");
    if std::env::var_os("CARGO_FEATURE_METAL").is_some() {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
            panic!("the metal feature is supported only on macOS");
        }
        cc::Build::new()
            .cpp(true)
            .file("native/metal_bridge.mm")
            .flag("-fobjc-arc")
            .flag("-std=c++17")
            .compile("rust_pw_dft_metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}
