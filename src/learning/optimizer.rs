
pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }


    pub fn update_weights(
        &self,
        weights: &mut [f32],
        gradients: &[f32],
    ) {
        assert_eq!(
            weights.len(),
            gradients.len(),
            "Weights and gradients must have the same length"
        );

        for i in 0..weights.len() {
            weights[i] -= self.learning_rate * gradients[i];
        }
    }

    pub fn update_bias(
        &self,
        bias: &mut f32,
        gradient: f32,
    ) {
        *bias -= self.learning_rate * gradient;
    }

    
}
