pub struct Activacion;

impl Activacion {
    pub fn relu(value: f32) -> f32 {
        if value > 0.0 { value } else { 0.0 }
    }

    pub fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }
}
