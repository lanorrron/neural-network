use std::ops::{Add, Mul, Sub};

#[derive(Debug)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Vector<T> {

    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }


    pub fn len(&self) -> usize {
        self.data.len()
    }


    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }


    pub fn get(&self, index: usize) -> &T {
        &self.data[index]
    }


    pub fn set(&mut self, index: usize, value: T) {
        self.data[index] = value;
    }
}


impl<T: Copy + Default> Vector<T> {
    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![T::default(); size],
        }
    }
}


impl<T: Copy + Add<Output = T>> Vector<T> {
    pub fn add(&self, other: &Vector<T>) -> Vector<T> {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        let mut data = Vec::new();

        for i in 0..self.len() {
            data.push(self.data[i] + other.data[i]);
        }

        Vector::new(data)
    }
}


impl<T: Copy + Sub<Output = T>> Vector<T> {
    pub fn sub(&self, other: &Vector<T>) -> Vector<T> {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        let mut data = Vec::new();

        for i in 0..self.len() {
            data.push(self.data[i] - other.data[i]);
        }

        Vector::new(data)
    }
}


impl<T: Copy + Mul<Output = T>> Vector<T> {
    pub fn multiply_scalar(&self, scalar: T) -> Vector<T> {
        let mut data = Vec::new();

        for i in 0..self.len() {
            data.push(self.data[i] * scalar);
        }

        Vector::new(data)
    }
}


impl<T> Vector<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    pub fn dot(&self, other: &Vector<T>) -> T {
        assert_eq!(self.len(), other.len(), "Vector dimensions must match");

        let mut result = T::default();

        for i in 0..self.len() {
            result = result + self.data[i] * other.data[i];
        }

        result
    }
}

impl<T: Copy> Vector<T> {
    pub fn map<F>(&self, mut function: F) -> Vector<T>
    where
        F: FnMut(T) -> T,
    {
        let mut data = Vec::new();

        for i in 0..self.len() {
            data.push(function(self.data[i]));
        }

        Vector::new(data)
    }
}

impl Vector<f32> {
    pub fn random(size: usize) -> Self {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut data = Vec::new();

        for _ in 0..size {
            data.push(rng.gen_range(-1.0..1.0));
        }

        Self::new(data)
    }
}
