use core::f64;
use kernel_density_estimation::prelude::*;


pub fn entropy(data: Vec<f64>, data_type: Option<&str>, bin_size: Option<f64>) -> f64 {
    // argument handling
    let data_type: &str = data_type.unwrap_or("kde");
    if data.len() == 1 {
        return 0.0
    }
    let length = data.len() as f64;
    let dev = std(&data);
    
    // Function Logic
    match data_type {
        "data" => {
            let bin_size = bin_size.unwrap_or(3.49 as f64 * dev.unwrap() * length.powf(-1.0/3.0)); // Scott 1979
            let bins = calc_bins(min(&data), max(&data), bin_size);
            let mut counts: Vec<u64> = bin_counts(&data, bins);
            counts.retain(|&x| x !=0);
            let sum: u64 = counts.iter().sum();
            let probability: Vec<f64> = counts.iter_mut().map(|x| *x as f64/sum as f64).collect();
            let entropy: f64 = -probability.iter().map(|&x| x * x.log2()).sum::<f64>();
            return entropy
        }
        "kde" => {
            let observations = data.to_vec();
            let bandwidth = Scott;
            let kernel = Epanechnikov;
            let kde = KernelDensityEstimator::new(observations, bandwidth, kernel);
            let pdf_max = (max(&data) / 0.1 + 1.0).ceil() as i32;
            let pdf_min = (min(&data) / 0.1).floor() as i32;
            let pdf_dataset: Vec<f64> = (pdf_min..pdf_max).into_iter().map(|x| x as f64 * 0.1).collect();

            // Sample the distribution.
            let histvals = kde.sample(pdf_dataset.as_slice(), 10_000);

            let bin_size = (pdf_max as f64 - pdf_min as f64) / 100.0;
            let bins = calc_bins(min(&histvals), max(&histvals), bin_size);
            let mut counts: Vec<u64> = bin_counts(&histvals, bins);
            counts.retain(|&x| x !=0);
            let sum: u64 = counts.iter().sum();
            let probability: Vec<f64> = counts.iter_mut().map(|x| *x as f64/sum as f64).collect();
            let entropy: f64 = -probability.iter().map(|&x| x * x.log2()).sum::<f64>();
            return entropy
        }
        _ => {
            println!("Unknown data_type");
            return 0.0
        }
    }
}




// Helper Functions
fn std(data: &Vec<f64>) -> Option<f64> {
    match (mean(data), data.len()) {
        (data_mean, count) if count > 0 => {
            let variance = data.iter().map(|value| {
                let diff = data_mean - (*value as f64);

                diff * diff
            }).sum::<f64>() / count as f64;

            Some(variance.sqrt())
        },
        _ => None
    }
}

fn mean(data: &Vec<f64>) -> f64 {
    let sum: f64 = data.iter().sum();
    sum as f64 / data.len() as f64
}

pub fn calc_bins(min: f64, max: f64, bin_size: f64) -> Vec<f64> {
    let array_len = ((max - min) / bin_size).ceil() as usize;  // Use usize for array length
    let mut bins = Vec::with_capacity(array_len);  // Create a Vec with capacity for `array_len`

    for index in 0..array_len+1 {
        if index ==0 {
            bins.push((bin_size * index as f64) -0.001 + min); // edge case: first bin wont include smallest value
        } else {
            bins.push(bin_size * index as f64 + min);  // Push the calculated bin value
        }
        
    }

    bins  // Return the Vec
}

fn bin_counts(data: &Vec<f64>, bins: Vec<f64>) -> Vec<u64> {
    let mut counts: Vec<u64> = Vec::with_capacity(bins.len()-1);
    for i in 0..bins.len() - 1 {
        let mn = bins[i];
        let mx = bins[i+1];
        counts.push(data.iter().map(|&x| if x>mn && x<=mx {1} else {0}).sum())
    }
    counts
}


pub fn max(arr: &Vec<f64>) -> f64 {
    let result = arr.iter().copied().fold(f64::NAN, f64::max);
    return result
}

pub fn min(arr: &Vec<f64>) -> f64 {
    let result = arr.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    return result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let data = [9.8, 7.4, 1.5, 3.0, 9.2, 6.0, 7.5, 10.1, 9.0, 10.5].to_vec();
        let max_test: f64 = max(&vec![0.0, 1.0]);
        assert_eq!(max, 1.0);
        let min_test: f64 = min(&vec![0.0, 1.0]);
        assert_eq!(max_test, 0.0);
        let mean_test = mean(&vec![1.0,2.0,3.0]);
        assert_eq!(mean_test, 2.0);
        let bins_test: Vec<f64> = calc_bins(1.5, 10.5, 1.5);
        assert_eq!(bins_test, [1.499, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5]);
        let result2: Vec<u64> = bin_counts(&data, bins);
        assert_eq!(result2, [2, 0, 1, 2, 1, 4]);
    }
}
