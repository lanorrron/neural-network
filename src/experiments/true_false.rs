use crate::learning::optimizer::GradientDescent;
use crate::math::vector::Vector;
use crate::neural::{activation::Activacion, neuron::Neuron};
use std::{io::{self, Write}, thread, time::Duration};

const EPOCH_DELAY: Duration = Duration::from_millis(40);
const LEARNING_RATE: f32 = 0.5;

/// Entrena una neurona para clasificar 0 como falso y 1 como verdadero.
pub fn run() {
    let samples = [(0.0_f32, 0.0_f32), (1.0, 1.0)];
    let mut neuron = Neuron::new(Vector::zeros(1), 0.0);
    let optimizer = GradientDescent::new(LEARNING_RATE);
    let max_epochs = 250;

    println!("Experimento: clasificar falso (0) y verdadero (1)");
    println!("La neurona usa sigmoid y el umbral de clasificación es 0.5.");
    println!("Cada línea muestra una actualización de descenso por gradiente.\n");

    for epoch in 1..=max_epochs {
        let mut weight_gradient = 0.0;
        let mut bias_gradient = 0.0;

        for &(input, target) in &samples {
            let logit = neuron.forward(&Vector::new(vec![input]));
            let probability = Activacion::sigmoid(logit);
            let error = (probability - target) / samples.len() as f32;
            weight_gradient += error * input;
            bias_gradient += error;
        }

        let old_weight = *neuron.weights().get(0);
        let old_bias = neuron.bias();
        let weight_delta = -LEARNING_RATE * weight_gradient;
        let bias_delta = -LEARNING_RATE * bias_gradient;
        optimizer.update_weights(
            neuron.weights_mut().as_mut_slice(),
            &[weight_gradient],
        );
        optimizer.update_bias(neuron.bias_mut(), bias_gradient);

        println!(
            "Época {epoch:>3} | ∇w={weight_gradient:+.6} ∇b={bias_gradient:+.6} | w: {old_weight:+.6} → {new_weight:+.6} (Δ {weight_delta:+.6}) | b: {old_bias:+.6} → {new_bias:+.6} (Δ {bias_delta:+.6})",
            new_weight = neuron.weights().get(0),
            new_bias = neuron.bias(),
        );
        let _ = io::stdout().flush();
        thread::sleep(EPOCH_DELAY);
    }

    println!("\nPredicciones finales:");
    for &(input, target) in &samples {
        let probability = Activacion::sigmoid(neuron.forward(&Vector::new(vec![input])));
        let prediction = probability >= 0.5;
        println!(
            "Entrada: {} | esperado: {} | probabilidad de verdadero: {:.4} | clasificado: {}",
            input as u8,
            label(target),
            probability,
            if prediction { "verdadero" } else { "falso" }
        );
    }
}

fn label(value: f32) -> &'static str {
    if value >= 0.5 { "verdadero" } else { "falso" }
}
