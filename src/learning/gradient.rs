pub struct Gradient;

impl Gradient {
    pub fn weights(loss_gradient: f32, inputs: &[f32]) -> Vec<f32> {
        let mut gradients = Vec::new();

        for input in inputs {
            let gradient = loss_gradient * input;
            gradients.push(gradient)
        }
        gradients
    }

    pub fn bias(loss_gradient: f32) -> f32 {
        loss_gradient
    }
}
