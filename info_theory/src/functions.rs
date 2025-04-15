use core::f64;
use kernel_density_estimation::prelude::*;
use std::collections::HashMap;


pub fn _entropy(data: Vec<f64>, data_type: Option<&str>, bin_size: Option<f64>) -> f64 {
    // argument handling
    let data_type: &str = data_type.unwrap_or("kde");
    if data.len() == 1 {
        return 0.0
    }
    
    // Function Logic
    match data_type {
        "data" => {
            // Direct calculation on the data using Scotts rule to determine bins
            // let bin_size = bin_size.unwrap_or(3.49 as f64 * dev.unwrap() * length.powf(-1.0/3.0)); // Scott 1979
            let bin_size = bin_size.unwrap_or(calc_bin_width_fd(&data));
            let bins = calc_bins(min(&data), max(&data), bin_size);
            let mut counts: Vec<u64> = bin_counts(&data, &bins);
            counts.retain(|&x| x !=0);
            let sum: u64 = counts.iter().sum();
            let probability: Vec<f64> = counts.iter_mut().map(|x| *x as f64/sum as f64).collect();
            let entropy: f64 = -probability.iter().map(|&x| x * x.log2()).sum::<f64>();
            return entropy
        }
        "kde" => {
            // Kernel Density Estimation (kde)
            let histvals = kde_sample(&data, "pdf");
            let bin_size = calc_bin_width_fd(&histvals);
            let bins = calc_bins(min(&histvals), max(&histvals), bin_size);
            let mut counts: Vec<u64> = bin_counts(&histvals, &bins);
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

pub fn _mutual_information(x_dat: Vec<f64>, y_dat: Vec<f64>, calc_type: Option<&str>) -> f64 {
    // Sanity check
    if x_dat.len() != y_dat.len() {
        panic!("x and y must be paired observations")
    }
    let calc_type: &str = calc_type.unwrap_or("data");
    let (x, y) = match calc_type {
        "data" => {
            // Use the actual data
            let x: Vec<f64> = x_dat.clone();
            let y: Vec<f64> = y_dat.clone();
            (x, y)
        }
        "kde" => {
            // Sample from the Kernel PDF
            // let x: Vec<f64> = kde_sample(&x_dat, "pdf").iter().copied().filter(|v| !v.is_nan()).collect();
            // let y: Vec<f64> = kde_sample(&y_dat, "pdf").iter().copied().filter(|v| !v.is_nan()).collect();
            let x: Vec<f64> = kde_sample(&x_dat, "pdf");
            let y: Vec<f64> = kde_sample(&y_dat, "pdf");
            (x, y)
        }
        _ => {
            println!("Unknown calc_type: Choose from [kde, data]");
            (vec![], vec![])
        }
    };

    let x_bins: Vec<f64> = calc_bins(min(&x), max(&x), calc_bin_width_fd(&x));
    let y_bins: Vec<f64> = calc_bins(min(&y), max(&y), calc_bin_width_fd(&y));
    // Calculate marginal distributions
    //  - probability of observing the data in x and y independent of each other
    let x_counts: Vec<u64> = bin_counts(&x, &x_bins);
    let x_sum = x_counts.iter().sum::<u64>();
    let marginal_x: Vec<f64> = x_counts.iter().map(|count| *count as f64 / x_sum as f64).collect();

    let y_counts: Vec<u64> = bin_counts(&y, &y_bins);
    let y_sum = y_counts.iter().sum::<u64>();
    let marginal_y: Vec<f64> = y_counts.iter().map(|count| *count as f64 / y_sum as f64).collect();
    // Calculate the joint distribution
    //  - probability of observing data in y given x
    //  - the same as observing data in x given y
    let in_bin_x: Vec<usize> = which_bin(&x, &x_bins);
    let in_bin_y: Vec<usize> = which_bin(&y, &y_bins);
    let joint = joint_pmf(in_bin_x, in_bin_y);
    // Calculate Mutual Information
    let mut mutual_info: f64 = 0.0;
    for (xi, row) in joint.iter().enumerate() {
        for (yi, &p_xy) in row.iter().enumerate() {
            let p_x = marginal_x[xi];
            let p_y = marginal_y[yi];
            // println!("p_x: {} | p_y: {} | p_xy: {} | in_log: {} | log: {}",p_x, p_y, p_xy, p_xy/(p_x*p_y), f64::log2(p_xy/(p_x*p_y)));
    
            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mutual_info += p_xy * f64::log2(p_xy / (p_x * p_y));
            }
        }
    }
    return mutual_info
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

fn bin_counts(data: &Vec<f64>, bins: &Vec<f64>) -> Vec<u64> {
    let mut counts: Vec<u64> = Vec::with_capacity(bins.len()-1);
    for i in 0..bins.len() - 1 {
        let mn = bins[i];
        let mx = bins[i+1];
        counts.push(data.iter().map(|&x| if x>mn && x<=mx {1} else {0}).sum())
    }
    counts
}

fn which_bin(data: &Vec<f64>, bins: &Vec<f64>) -> Vec<usize> {
    // Takes in a data vector and bin edges
    // returns a vector of the same size as data where the values are
    // the indices of the bin that the data point falls into
    data.iter().map(|&x| {
        match bins.windows(2).position(|w| x >= w[0] && x < w[1]) {
            Some(idx) => idx,
            None => panic!("Data outside of provided bins")
        }
    }).collect()

}

fn joint_pmf(x: Vec<usize>,y: Vec<usize>) -> Vec<Vec<f64>>{
    // x and y are vectors of bin indices 
    //    eg. [0, 1, 1]
    //      - 1st datapoint is in bin 0
    //      - 2nd and 3rd data points are in bin 1
    let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        *joint_counts.entry((xi, yi)).or_insert(0) += 1;

    }
    let y_max = max(&y.iter().map(|&y| y as f64).collect()) as usize;
    let x_max = max(&x.iter().map(|&x| x as f64).collect()) as usize;
    let mut joint_pmf = vec![vec![0.0; y_max + 1]; x_max + 1];
    let total: f64 = joint_counts.values().map(|&count| count as f64).sum();
    for ((xi, yi), count) in joint_counts {
        joint_pmf[xi][yi] = count as f64 / total;
    }
    joint_pmf

}

fn kde_sample(data: &Vec<f64>, return_type: &str) -> Vec<f64>{
    let observations = data.clone();
    let bandwidth = Scott;
    let kernel = Epanechnikov;
    let kde = KernelDensityEstimator::new(observations, bandwidth, kernel);
    let pdf_max = (max(&data) / 0.1 + 1.0).ceil() as i32;
    let pdf_min = (min(&data) / 0.1).floor() as i32;
    let pdf_dataset: Vec<f64>  = (pdf_min..pdf_max).into_iter().map(|x| x as f64 * 0.1).collect();
    let values = match return_type {
        "pdf" => kde.pdf(pdf_dataset.as_slice()),
        "samples" => kde.sample(pdf_dataset.as_slice(), 10_000),
        _ => panic!("kde_sample Error. Uknown return_type. Choose from: pdf, samples")
    };
    return values
}

pub fn max(arr: &Vec<f64>) -> f64 {
    let result = arr.iter().copied().fold(f64::NAN, f64::max);
    return result
}

pub fn min(arr: &Vec<f64>) -> f64 {
    let result = arr.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    return result
}

fn calc_bin_width_fd(data: &Vec<f64>) -> f64 {
    // Calculate bin width using the Freedman-Diaconis rule
    // Sort the data
    let mut sorted: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Helper function to find the percentile
    let percentile = |p: f64| -> f64 {
        let idx = (p * (sorted.len() as f64 - 1.0)).round() as usize;
        sorted[idx]
    };
    // Compute Q1 and Q3
    let q1 = percentile(0.25);
    let q3 = percentile(0.75);
    // Return the IQR
    let iqr = q3 - q1;
    let length = sorted.len() as f64;
    (2.0 * iqr) / length.powf(1.0/3.0)
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
