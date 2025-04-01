use kernel_density_estimation::prelude::*;
use plotly::color::NamedColor;
use plotly::common::{Marker, Mode, AxisSide};
use plotly::{Histogram, Plot, Scatter, layout::{Layout, Axis}};
use crate::functions::{max, min};

pub fn kde_plot(data: Vec<f64>) {
    let observations = data.clone();
    let bandwidth = Scott;
    let kernel = Epanechnikov;
    let x1 = observations.clone();
    let y1 = vec![0.0; data.len()];
    let kde = KernelDensityEstimator::new(observations, bandwidth, kernel);
    let pdf_max = (max(&data) / 0.1 + 1.0).ceil() as i32;
    let pdf_min = (min(&data) / 0.1).floor() as i32;
    let pdf_dataset: Vec<f64> = (pdf_min..pdf_max).into_iter().map(|x| x as f64 * 0.1).collect();
    let cdf_dataset = pdf_dataset.clone();
    let sample_dataset = cdf_dataset.clone();

    // Evaluate the PDF.
    let x2 = pdf_dataset.clone();
    let y2 = kde.pdf(pdf_dataset.as_slice());

    // Evaluate the CDF.
    let x3 = cdf_dataset.clone();
    let y3 = kde.cdf(cdf_dataset.as_slice());

    // Sample the distribution.
    let x4 = kde.sample(sample_dataset.as_slice(), 10_000);

    // Plot the observations.
    let trace1 = Scatter::new(x1, y1)
        .mode(Mode::Markers)
        .marker(Marker::new().color(NamedColor::CornflowerBlue))
        .name("Data");
    // Plot the PDF.
    let trace2 = Scatter::new(x2, y2)
        .mode(Mode::Lines)
        .marker(Marker::new().color(NamedColor::Black))
        .name("PDF");
    // Plot the CDF.
    let trace3 = Scatter::new(x3, y3)
        .mode(Mode::Lines)
        .marker(Marker::new().color(NamedColor::YellowGreen))
        .name("CDF")
        .y_axis("y2");
    // Plot the samples as a histogram.
    let trace4 = Histogram::new(x4)
        .hist_norm(plotly::histogram::HistNorm::ProbabilityDensity)
        .marker(Marker::new().color(NamedColor::Bisque))
        .n_bins_x(100)
        .name("Histogram");

    let layout = Layout::new()
        .y_axis(Axis::new().title("P(A)")) // Main Y-axis
        .y_axis2(Axis::new()
            .title("CDF")
            .overlaying("y") // Overlay on the primary Y-axis
            .side(AxisSide::Right)); // Place on the right side

    // Render the plot.
    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);
    plot.add_trace(trace4);
    plot.set_layout(layout);
    plot.show();
}