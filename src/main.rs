use info_theory::functions::*;
use info_theory::plots::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};



fn main() {
    let ctype = Some("data");
    let x = [9.8, 7.4, 1.5, 3.0, 9.2, 6.0, 7.5, 10.1, 9.0, 10.5].to_vec();
    let y = [9.8, 7.4, 1.5, 3.0, 9.2, 6.0, 7.5, 10.1, 9.0, 10.5].to_vec();
    // let y = [1.8, 1.4, 2.5, 3.0, 1.2, 2.0, 2.5, 0.1, 1.9, 2.3].to_vec();
    let result = _mutual_information(x.clone(),y.clone(), ctype);
    println!("{}", result);
    let x_ent = _entropy(x.clone(), ctype, None);
    let y_ent = _entropy(y.clone(), ctype, None);
    println!("x-entropy: {}", x_ent);
    println!("y-entropy: {}", y_ent);
    println!("Bounded: {}", (2.0*result)/(x_ent+y_ent))

}

fn main_entropy() {
    let data = [9.8, 7.4, 1.5, 3.0, 9.2, 6.0, 7.5, 10.1, 9.0, 10.5].to_vec();
    let result = _entropy(data.clone(), Some("data"), Some(1.5));
    println!("Data-based Entropy: {}  |  Bin Size: 1.5", result);
    let kde: f64 = _entropy(data.clone(), Some("kde"), None);
    println!("KDE-based Entropy: {}", kde);

    // Uniform Distribution Testing
    let mut rng = rand::thread_rng();
    let size = 1000; // Number of random values
    let range = 0.0..100.0; // Define the range
    let uniform_vals: Vec<f64> = (0..size)
        .map(|_| rng.gen_range(range.clone())) // Generate f64 in range
        .collect();
    println!("Uniform distrobution: {}", _entropy(uniform_vals.clone(), Some("kde"), None));
    kde_plot(uniform_vals);

    // Normal Distribution Testing
    let mut rng = rand::thread_rng();
    let size = 1000; // Number of random values
    let normal = Normal::new(0.0, 2.0).expect("Failed to create normal distribution"); // Mean = 0.0, Std Dev = 1.0
    let normal_vals: Vec<f64> = (0..size) // Generate 10 samples
        .map(|_| normal.sample(&mut rng))
        .collect();
    println!("Normal distrobution: {}", _entropy(normal_vals.clone(), Some("kde"), None));
    kde_plot(normal_vals);
}