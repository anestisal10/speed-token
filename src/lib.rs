use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::ffi::{CString};
use libc::{c_char, size_t};

// Opaque struct for C's Tokenizer
#[repr(C)]
pub struct CTokenizer { _unused: [u8; 0] }

unsafe extern "C" {
    fn tokenizer_load(path: *const c_char, pattern: *const c_char) -> *mut CTokenizer;
    fn tokenizer_free(t: *mut CTokenizer);
    fn tokenizer_encode(t: *mut CTokenizer, text: *const c_char, out_tokens: *mut i32, max_tokens: size_t) -> size_t;
    fn tokenizer_decode(t: *mut CTokenizer, tokens: *const i32, num_tokens: size_t, out_text: *mut c_char, max_text_len: size_t) -> size_t;
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

    pub fn encode(&self, text: &str) -> Vec<i32> {
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

    pub fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<i32>> {
        texts.into_par_iter()
            .map(|text| self.encode(&text))
            .collect()
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
