fn main() {
    // On macOS, extension modules must leave libpython symbols undefined
    // (`-undefined dynamic_lookup`) so that plain `cargo build` links.
    pyo3_build_config::add_extension_module_link_args();
}
