pub struct MSE;

impl MSE {
    pub fn calculate(prediction: &[f32], target: &[f32]) -> f32 {
        assert_eq!(
            prediction.len(),
            target.len(),
            "prediction and target must have the same length"
        );

        assert!(!prediction.is_empty(), "Vectors cannot be empty");

        let mut sum = 0.0;

        for i in 0..prediction.len() {
            let error = prediction[i] - target[i];
            sum += error * error;
        }
        sum / prediction.len() as f32
    }

    pub fn derivate(prediction: &[f32], target: &[f32]) -> Vec<f32> {
           assert_eq!(
            prediction.len(),
            target.len(),
            "Prediction and target must have the same length"
        );

        assert!(!prediction.is_empty(), "Vectors cannot be empty");

        let n = prediction.len() as f32;
        let mut gradients = Vec::new();

        for i in 0..prediction.len(){
            let gradient = 2.0 * (prediction[i] - target[i]) / n;

            gradients.push(gradient)
        }
        gradients
    }
}
