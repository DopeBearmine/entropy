use pyo3::prelude::*;
use info_theory::functions::*;


#[pyfunction]
fn entropy (data: Vec<f64>, data_type: Option<&str>, bin_size: Option<f64>) -> PyResult<f64> {
    Ok(_entropy(data, data_type, bin_size))
}

#[pymodule]
fn information_theory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(entropy, m)?)?;
    Ok(())
}
