
use crate::math::vector::Vector;

pub struct Neuron {
    weights: Vector<f32>,
    bias: f32,
}

impl Neuron {
    pub fn new(weights: Vector<f32>, bias: f32) -> Self {
        Self {
            weights,
            bias,
        }
    }

    pub fn forward(&self, inputs: &Vector<f32>) -> f32 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Input and weights dimensions must match"
        );

        inputs.dot(&self.weights) + self.bias
    }

    pub fn weights_mut(&mut self) -> &mut Vector<f32> {
        &mut self.weights
    }

    pub fn weights(&self) -> &Vector<f32> {
        &self.weights
    }

    pub fn bias_mut(&mut self) -> &mut f32 {
        &mut self.bias
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }
}
