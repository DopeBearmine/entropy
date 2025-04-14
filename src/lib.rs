use pyo3::prelude::*;
use info_theory::functions::*;


#[pyfunction]
fn entropy (data: Vec<f64>, data_type: Option<&str>, bin_size: Option<f64>) -> PyResult<f64> {
    Ok(_entropy(data, data_type, bin_size))
}

#[pyfunction]
fn mutual_information(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    Ok(_mutual_information(x, y))
}

#[pymodule]
fn information_theory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(entropy, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information, m)?)?;
    Ok(())
}
