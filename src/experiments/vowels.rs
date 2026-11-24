use crate::learning::{gradient::Gradient, optimizer::GradientDescent};
use crate::math::vector::Vector;
use crate::neural::{layer::Layer, neuron::Neuron};
use std::{
    io::{self, Write},
    thread,
    time::Duration,
};

const VOCALS: [char; 5] = ['A', 'E', 'I', 'O', 'U'];
const PIXELS: usize = 25;
const LEARNING_RATE: f32 = 0.3;
const EPOCHS: usize = 500;
const REPORT_EVERY: usize = 1;

// Cada vocal es un patrón binario de 5 x 5. '#' representa un píxel encendido.
const PATTERNS: [[&str; 5]; 5] = [
    [".###.", "#...#", "#####", "#...#", "#...#"], // A
    ["#####", "#....", "####.", "#....", "#####"], // E
    ["#####", "..#..", "..#..", "..#..", "#####"], // I
    [".###.", "#...#", "#...#", "#...#", ".###."], // O
    ["#...#", "#...#", "#...#", "#...#", ".###."], // U
];

/// Entrena una capa de cinco neuronas, una salida por vocal.
pub fn run(input_pattern: Option<&str>) {
    let samples: Vec<Vector<f32>> = PATTERNS.iter().map(encode).collect();
    let neurons = (0..VOCALS.len())
        .map(|_| Neuron::new(Vector::zeros(PIXELS), 0.0))
        .collect();
    let mut layer = Layer::new(neurons);
    let optimizer = GradientDescent::new(LEARNING_RATE);

    println!("Experimento: reconocer las vocales A, E, I, O y U");
    println!("Cada vocal es un dibujo binario de 5 x 5 píxeles.");
    println!("La capa contiene 5 neuronas; cada una recibe los 25 píxeles.");
    println!("Para probar una vocal: cargo run -- a\n");

    for epoch in 1..=EPOCHS {
        let mut weight_gradients = vec![vec![0.0_f32; PIXELS]; VOCALS.len()];
        let mut bias_gradients = vec![0.0_f32; VOCALS.len()];

        // Acumulamos los gradientes del conjunto y actualizamos después de recorrerlo.
        for (sample_index, sample) in samples.iter().enumerate() {
            let logits = layer.forward(sample);
            let probabilities = softmax(&logits);

            for output in 0..VOCALS.len() {
                let target = if output == sample_index { 1.0 } else { 0.0 };
                let error = (probabilities.as_slice()[output] - target) / samples.len() as f32;
                let sample_weight_gradients = Gradient::weights(error, sample.as_slice());

                for (total, sample_gradient) in weight_gradients[output]
                    .iter_mut()
                    .zip(sample_weight_gradients)
                {
                    *total += sample_gradient;
                }
                bias_gradients[output] += Gradient::bias(error);
            }
        }

        let mut largest_change = 0.0_f32;
        for (output, neuron) in layer.neurons_mut().iter_mut().enumerate() {
            for gradient in &weight_gradients[output] {
                largest_change = largest_change.max((LEARNING_RATE * gradient).abs());
            }
            largest_change = largest_change.max((LEARNING_RATE * bias_gradients[output]).abs());
            optimizer.update_weights(
                neuron.weights_mut().as_mut_slice(),
                &weight_gradients[output],
            );
            optimizer.update_bias(neuron.bias_mut(), bias_gradients[output]);
        }

        if epoch % REPORT_EVERY == 0 || epoch == 1 {
            println!(
                "Época {epoch:>3}/{EPOCHS} | cambio máximo en pesos/bias: {largest_change:.6}"
            );
            print_predictions(&layer, &samples);
            let _ = io::stdout().flush();
            thread::sleep(Duration::from_millis(80));
        }
    }

    if let Some(vowel_text) = input_pattern {
        match parse_vowel(vowel_text) {
            Ok(vowel_index) => {
                let pattern = &PATTERNS[vowel_index];
                let input = encode(pattern);
                let probabilities = softmax(&layer.forward(&input));
                let predicted = argmax(&probabilities);
                println!("\nVocal recibida: {}", VOCALS[vowel_index]);
                println!("Patrón usado internamente:");
                for row in pattern {
                    println!("  {row}");
                }
                println!("La red reconoce la vocal: {}", VOCALS[predicted]);
                println!("Probabilidades: ");
                for output in 0..VOCALS.len() {
                    println!(
                        "  {}: {:.2}%",
                        VOCALS[output],
                        probabilities.as_slice()[output] * 100.0
                    );
                }
            }
            Err(message) => eprintln!("Patrón inválido: {message}"),
        }
    } else {
        println!("\nReconocimiento final de los patrones aprendidos:");
        for (expected, sample) in samples.iter().enumerate() {
            let probabilities = softmax(&layer.forward(sample));
            let predicted = argmax(&probabilities);
            println!(
                "Esperada: {} | reconocida: {} | confianza: {:.2}%",
                VOCALS[expected],
                VOCALS[predicted],
                probabilities.as_slice()[predicted] * 100.0
            );
        }
    }
}

fn encode(pattern: &[&str; 5]) -> Vector<f32> {
    let pixels = pattern
        .iter()
        .flat_map(|row| {
            row.chars()
                .map(|pixel| if pixel == '#' { 1.0 } else { 0.0 })
        })
        .collect();
    Vector::new(pixels)
}

fn parse_vowel(value: &str) -> Result<usize, &'static str> {
    if value.chars().count() != 1 {
        return Err("pasa una sola vocal: a, e, i, o o u");
    }
    match value.to_ascii_uppercase().as_str() {
        "A" => Ok(0),
        "E" => Ok(1),
        "I" => Ok(2),
        "O" => Ok(3),
        "U" => Ok(4),
        _ => Err("la entrada debe ser una vocal: a, e, i, o o u"),
    }
}

fn softmax(logits: &Vector<f32>) -> Vector<f32> {
    let max_logit = logits
        .as_slice()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = logits
        .as_slice()
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect();
    let sum: f32 = exponentials.iter().sum();
    Vector::new(exponentials.iter().map(|value| value / sum).collect())
}

fn argmax(values: &Vector<f32>) -> usize {
    values
        .as_slice()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn print_predictions(layer: &Layer, samples: &[Vector<f32>]) {
    for (expected, sample) in samples.iter().enumerate() {
        let probabilities = softmax(&layer.forward(sample));
        let predicted = argmax(&probabilities);
        print!("  {} → {} [", VOCALS[expected], VOCALS[predicted]);
        for output in 0..VOCALS.len() {
            if output > 0 {
                print!(" ");
            }
            print!("{}:{:.2}", VOCALS[output], probabilities.as_slice()[output]);
        }
        println!("]");
    }
}
