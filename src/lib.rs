use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::ffi::{CString};
use libc::{c_char, size_t};

// Opaque struct for C's Tokenizer
#[repr(C)]
pub struct CTokenizer { _unused: [u8; 0] }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CTokenizerProfile {
    pub pretokenize_cycles: u64,
    pub merge_cycles: u64,
    pub hash_lookup_cycles: u64,
    pub total_cycles: u64,
}

unsafe extern "C" {
    fn tokenizer_load(path: *const c_char, pattern: *const c_char) -> *mut CTokenizer;
    fn tokenizer_free(t: *mut CTokenizer);
    fn tokenizer_encode(t: *mut CTokenizer, text: *const c_char, out_tokens: *mut i32, max_tokens: size_t) -> size_t;
    fn tokenizer_decode(t: *mut CTokenizer, tokens: *const i32, num_tokens: size_t, out_text: *mut c_char, max_text_len: size_t) -> size_t;
    fn tokenizer_thread_free();
    fn tokenizer_reset_profile(t: *mut CTokenizer);
    fn tokenizer_get_profile(t: *mut CTokenizer) -> CTokenizerProfile;
}

#[pyclass]
pub struct SpeedToken {
    tokenizer: *mut CTokenizer,
}

unsafe impl Send for SpeedToken {}
unsafe impl Sync for SpeedToken {}

#[pymethods]
impl SpeedToken {
    #[new]
    #[pyo3(signature = (path, pattern = "cl100k_base"))]
    pub fn new(path: &str, pattern: &str) -> PyResult<Self> {
        let path_c = CString::new(path).map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid path"))?;
        let pattern_c = CString::new(pattern).map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid pattern"))?;
        let t = unsafe { tokenizer_load(path_c.as_ptr(), pattern_c.as_ptr()) };
        if t.is_null() {
            return Err(pyo3::exceptions::PyIOError::new_err("Failed to load tokenizer"));
        }
        Ok(SpeedToken { tokenizer: t })
    }

    pub fn encode(&self, text_obj: &pyo3::Bound<'_, pyo3::types::PyString>) -> Vec<i32> {
        let text_str = match text_obj.to_str() {
            Ok(s) => {
                s.to_string()
            },
            Err(_) => {
                let py = text_obj.py();
                let locals = pyo3::types::PyDict::new(py);
                locals.set_item("text", text_obj).unwrap();
                let eval_code = std::ffi::CStr::from_bytes_with_nul(
                    b"\"\".join([c if not (0xD800 <= ord(c) <= 0xDFFF) else chr(0xFFFD) for c in text])\0"
                ).unwrap();
                let replaced = py.eval(
                    &*eval_code,
                    None,
                    Some(&locals),
                ).unwrap();
                let s = replaced.extract::<String>().unwrap();
                s
            }
        };
        let text = text_str.as_str();
        if text.is_empty() {
            return Vec::new();
        }
        let text_c = CString::new(text).unwrap_or_else(|_| CString::new("").unwrap());
        // Max tokens can't exceed bytes (worst case 1 byte -> 1 token)
        let max_tokens = text.len() + 1;
        let mut tokens = vec![0i32; max_tokens];
        let n = unsafe { tokenizer_encode(self.tokenizer, text_c.as_ptr(), tokens.as_mut_ptr(), max_tokens) };
        tokens.truncate(n);
        tokens
    }

    pub fn decode(&self, py: Python, tokens: Vec<i32>) -> PyObject {
        if tokens.is_empty() {
            return PyBytes::new(py, &[]).into();
        }
        // Max bytes heuristic: 32 bytes per token is very safe for cl100k/o200k
        let max_text_len = tokens.len() * 32;
        let mut buffer = vec![0u8; max_text_len];
        let n = unsafe { tokenizer_decode(self.tokenizer, tokens.as_ptr(), tokens.len(), buffer.as_mut_ptr() as *mut c_char, max_text_len) };
        PyBytes::new(py, &buffer[..n]).into()
    }

    pub fn encode_batch(&self, texts: Vec<pyo3::Py<pyo3::types::PyString>>, py: Python) -> Vec<Vec<i32>> {
        // Since we need to access python methods, we should do the transformation centrally, 
        // parallelizing over the Rust strings.
        let rust_strings: Vec<String> = texts.into_iter().map(|t| {
            let bound = t.bind(py);
            match bound.to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    let locals = pyo3::types::PyDict::new(py);
                    locals.set_item("text", &bound).unwrap();
                    let eval_code = std::ffi::CStr::from_bytes_with_nul(
                        b"\"\".join([c if not (0xD800 <= ord(c) <= 0xDFFF) else chr(0xFFFD) for c in text])\0"
                    ).unwrap();
                    let replaced = py.eval(
                        &*eval_code,
                        None,
                        Some(&locals),
                    ).unwrap();
                    replaced.extract::<String>().unwrap()
                }
            }
        }).collect();

        rust_strings.into_par_iter()
            .map(|text| {
                if text.is_empty() {
                    return Vec::new();
                }
                let text_c = CString::new(text.as_str()).unwrap_or_else(|_| CString::new("").unwrap());
                let max_tokens = text.len() + 1;
                let mut tokens = vec![0i32; max_tokens];
                let n = unsafe { tokenizer_encode(self.tokenizer, text_c.as_ptr(), tokens.as_mut_ptr(), max_tokens) };
                tokens.truncate(n);
                tokens
            })
            .collect()
    }

    pub fn reset_profile(&self) {
        unsafe { tokenizer_reset_profile(self.tokenizer) };
    }

    pub fn get_profile(&self, py: Python) -> PyObject {
        let p = unsafe { tokenizer_get_profile(self.tokenizer) };
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("pretokenize_cycles", p.pretokenize_cycles).unwrap();
        dict.set_item("merge_cycles", p.merge_cycles).unwrap();
        dict.set_item("hash_lookup_cycles", p.hash_lookup_cycles).unwrap();
        dict.set_item("total_cycles", p.total_cycles).unwrap();
        dict.into()
    }

    #[staticmethod]
    pub fn clean_thread_locals() {
        unsafe { tokenizer_thread_free() };
    }
}

impl Drop for SpeedToken {
    fn drop(&mut self) {
        unsafe { tokenizer_free(self.tokenizer) };
    }
}

#[pymodule]
fn speed_token(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SpeedToken>()?;
    Ok(())
}
