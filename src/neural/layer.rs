use crate::math::vector::Vector;
use crate::neural::neuron::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    pub fn forward(&self, inputs: &Vector<f32>) -> Vector<f32> {
        let mut outputs = Vec::new();

        for neuron in &self.neurons {
            outputs.push(neuron.forward(inputs));
        }

        Vector::new(outputs)
    }

    pub fn neurons_mut(&mut self) -> &mut [Neuron] {
        &mut self.neurons
    }
}
