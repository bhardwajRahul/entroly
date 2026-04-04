---
claim_id: 18a33f4c10049eb001f62ab0
entity: depgraph
status: inferred
confidence: 0.75
sources:
  - depgraph.rs:32
  - depgraph.rs:44
  - depgraph.rs:61
  - depgraph.rs:74
  - depgraph.rs:84
  - depgraph.rs:89
  - depgraph.rs:99
  - depgraph.rs:167
  - depgraph.rs:456
  - depgraph.rs:596
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: depgraph

**Language:** rs
**Lines of code:** 1432

## Types
- `pub struct Dependency` — A directed dependency between two fragments.
- `pub enum DepType`
- `pub struct DepGraph` — The dependency graph.
- `pub struct Engine`

## Functions
- `pub fn new() -> Self`
- `pub fn register_symbol(&mut self, symbol: &str, fragment_id: &str)` — Register a symbol definition (e.g., function, class, type).
- `pub fn add_dependency(&mut self, dep: Dependency)` — Add a dependency edge.
- `pub fn auto_link(&mut self, fragment_id: &str, content: &str)` — Auto-link: given a fragment's content, extract symbols it references and create dependency edges to the fragments that define those symbols.
- `fn extract_import_targets(content: &str) -> HashSet<String>` — Extract symbols that are explicitly imported in source code. Parses Python, Rust, JS/TS, Go, Java, C/C++, and Ruby imports.
- `fn extract_ffi_exports(content: &str) -> Vec<(String, String)>` — Detects: - **PyO3**: `#[pyfunction] fn process()` → exports "process" - **PyO3**: `#[pyclass] struct Engine` → exports "Engine" - **JNI**: `JNIEXPORT ... Java_com_example_Class_method` → exports "meth
- `fn is_type_reference(lower_content: &str, ident: &str) -> bool` — Check if a symbol appears to be used as a type reference (e.g., `: TypeName`, `-> TypeName`, `<TypeName>`).
- `pub fn transitive_deps(&self, fragment_id: &str, max_depth: usize) -> Vec<String>` — Get all fragments that this fragment depends on (transitively).
- `pub fn reverse_deps(&self, fragment_id: &str) -> Vec<String>` — Get all fragments that DEPEND ON this fragment (reverse deps).
- `pub fn compute_dep_boosts(` — Compute dependency boost for each fragment.  If fragment A is selected and A depends on B, then B gets a boost proportional to the dependency strength.  This solves the "context is not additive" probl
- `pub fn connected_components(&self, fragment_ids: &[String]) -> Vec<Vec<String>>` — Find connected components — fragments that should be selected or dropped together.
- `pub fn symbol_definitions(&self) -> &HashMap<String, String>` — Get all symbol → fragment_id mappings.
- `pub fn has_symbol(&self, symbol: &str) -> bool` — Check if a symbol is defined anywhere.
- `pub fn edge_count(&self) -> usize`
- `pub fn node_count(&self) -> usize`
- `pub fn extract_identifiers(content: &str) -> Vec<String>` — Extract identifiers (function names, variable names) from source code. Fast, regex-free extraction for supported languages.
- `fn extract_definitions(content: &str) -> Vec<String>` — Extract symbol definitions (def, class, fn, struct, etc.)
- `fn extract_require_lhs(line: &str) -> Option<String>` — Extract the LHS variable name from `const foo = require('...')`.
- `fn extract_fn_name_from_line(line: &str) -> Option<String>` — Extract function name from a line like "fn process(" or "pub fn process(".
- `fn extract_struct_name_from_line(line: &str) -> Option<String>` — Extract struct/class name from a line like "struct Engine" or "pub struct Engine {".
- `fn is_keyword(word: &str) -> bool` — Check if an identifier is a language keyword (ignore these).
- `fn test_extract_definitions_python()`
- `fn test_extract_definitions_rust()`
- `fn test_dependency_boost()`
- `fn test_connected_components()`
- `fn test_transitive_deps()`
- `fn test_extract_definitions_js_function()`
- `fn test_extract_definitions_arrow_functions()`
- `fn test_extract_definitions_ts_interface_type_enum()`
- `fn test_extract_import_targets_default_import()`
- `fn test_extract_import_targets_require()`
- `fn test_pyo3_rust_to_python_linking()`
- `fn process_data(input: &str) -> PyResult<String>`
- `fn test_jni_java_to_c_linking()`
- `fn test_cgo_export_linking()`
- `fn test_wasm_bindgen_linking()`
- `pub fn greet(name: &str) -> String`
- `fn test_napi_linking()`
- `fn test_c_ffi_extern_c_linking()`
- `fn test_cross_lang_dep_boost()`
- `fn test_pymethods_exports_methods()`
- `fn run(&self, input: &str) -> String`
- `pub fn stop(&self)`
