use info_theory::functions::*;
use info_theory::plots::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
    let data = [9.8, 7.4, 1.5, 3.0, 9.2, 6.0, 7.5, 10.1, 9.0, 10.5].to_vec();
    let result = entropy(data.clone(), Some("data"), Some(1.5));
    println!("Data-based Entropy: {}  |  Bin Size: 1.5", result);
    let kde: f64 = entropy(data.clone(), Some("kde"), None);
    println!("KDE-based Entropy: {}", kde);

    // Uniform Distribution Testing
    let mut rng = rand::thread_rng();
    let size = 1000; // Number of random values
    let range = 0.0..100.0; // Define the range
    let uniform_vals: Vec<f64> = (0..size)
        .map(|_| rng.gen_range(range.clone())) // Generate f64 in range
        .collect();
    println!("Uniform distrobution: {}", entropy(uniform_vals.clone(), Some("kde"), None));
    kde_plot(uniform_vals);

    // Normal Distribution Testing
    let mut rng = rand::thread_rng();
    let size = 1000; // Number of random values
    let normal = Normal::new(0.0, 2.0).expect("Failed to create normal distribution"); // Mean = 0.0, Std Dev = 1.0
    let normal_vals: Vec<f64> = (0..size) // Generate 10 samples
        .map(|_| normal.sample(&mut rng))
        .collect();
    println!("Normal distrobution: {}", entropy(normal_vals.clone(), Some("kde"), None));
    kde_plot(normal_vals);
}