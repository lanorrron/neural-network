use crate::learning::{gradient::Gradient, loss::MSE, optimizer::GradientDescent};
use crate::math::vector::Vector;
use crate::neural::neuron::Neuron;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

const REPORT_EVERY: usize = 20;
const REPORT_DELAY: Duration = Duration::from_millis(150);

pub fn run() {
    // Experimento: aprender y = 2x con una neurona lineal.
    let inputs = [1.0, 2.0, 3.0, 4.0, 7.0, 8.0];
    let targets = [2.0, 4.0, 6.0, 8.0, 14.0, 16.0];
    let mut neuron = Neuron::new(Vector::zeros(1), 0.0);
    let optimizer = GradientDescent::new(0.02);
    let max_epochs = 10_000;
    let target_loss = 0.000_000_000_001;

    println!("Experimento: aprender y = 2x");
    println!("Cada {REPORT_EVERY} épocas se muestran parámetros y errores.\n");
    print_snapshot(0, &neuron, &inputs, &targets);

    let mut trained_epochs = 0;
    for epoch in 1..=max_epochs {
        trained_epochs = epoch;
        let mut weight_gradients = vec![0.0; 1];
        let mut bias_gradient = 0.0;

        for (&x, &target) in inputs.iter().zip(&targets) {
            let input = Vector::new(vec![x]);
            let prediction = neuron.forward(&input);
            let loss_gradient = MSE::derivate(&[prediction], &[target])[0]
                / inputs.len() as f32;
            let sample_weight_gradients = Gradient::weights(loss_gradient, &[x]);

            for (total, sample) in weight_gradients.iter_mut().zip(sample_weight_gradients) {
                *total += sample;
            }
            bias_gradient += Gradient::bias(loss_gradient);
        }

        optimizer.update_weights(neuron.weights_mut().as_mut_slice(), &weight_gradients);
        optimizer.update_bias(neuron.bias_mut(), bias_gradient);

        let loss = calculate_loss(&neuron, &inputs, &targets);
        let reached_target = loss <= target_loss;
        if epoch % REPORT_EVERY == 0 || reached_target {
            print_snapshot(epoch, &neuron, &inputs, &targets);
            thread::sleep(REPORT_DELAY);
        }
        if reached_target {
            break;
        }
    }

    let final_loss = calculate_loss(&neuron, &inputs, &targets);
    println!("Entrenamiento terminado en {trained_epochs} épocas.");
    println!("Meta alcanzada: {} (MSE = {final_loss:.15})", final_loss <= target_loss);
}

fn calculate_loss(neuron: &Neuron, inputs: &[f32], targets: &[f32]) -> f32 {
    let predictions: Vec<f32> = inputs
        .iter()
        .map(|&x| neuron.forward(&Vector::new(vec![x])))
        .collect();
    MSE::calculate(&predictions, targets)
}

fn print_snapshot(epoch: usize, neuron: &Neuron, inputs: &[f32], targets: &[f32]) {
    let loss = calculate_loss(neuron, inputs, targets);
    println!(
        "Época {epoch:>4} | peso = {:.7} | bias = {:.7} | MSE = {loss:.15}",
        neuron.weights().get(0),
        neuron.bias()
    );
    println!("  x | esperado | predicción | error (predicción - esperado)");

    for (&x, &target) in inputs.iter().zip(targets) {
        let prediction = neuron.forward(&Vector::new(vec![x]));
        let error = prediction - target;
        println!(" {x:.0} | {target:>8.3} | {prediction:>10.5} | {error:>+10.7}");
    }
    println!();
    let _ = io::stdout().flush();
}
