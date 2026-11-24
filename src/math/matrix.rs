use std::{ops::{Add, Mul, Sub}};

pub struct Matrix<T> {
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn new(rows: usize, columns: usize, data: Vec<T>) -> Self {
        assert_eq!(
            rows * columns,
            data.len(),
            "The number of elements does not match the dimensions"
        );

        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn get(&self, row: usize, column: usize) -> &T {
        assert!(row < self.rows, "Row index out of bounds");
        assert!(column < self.columns, "Column index out of bounds");

        let position = row * self.columns + column;
        &self.data[position]
    }

    pub fn set(&mut self, row: usize, column: usize, value: T) {
        assert!(row < self.rows, "Row index out of bounds");
        assert!(column < self.columns, "Column index out of bounds");

        let position = row * self.columns + column;
        self.data[position] = value;
    }
}

impl<T: Copy> Matrix<T> {
    pub fn transpose(&self) -> Matrix<T> {
        let mut data: Vec<T> = Vec::new();

        for i in 0..self.columns {
            for j in 0..self.rows {
                let position = j * self.columns + i;
                data.push(self.data[position]);
            }
        }

        Matrix::new(self.columns, self.rows, data)
    }
}

impl<T: std::fmt::Display> Matrix<T> {
    pub fn print(&self) {
        for row in 0..self.rows {
            print!("[");
            for column in 0..self.columns {
                if column > 0 {
                    print!(", ");
                }
                print!("{}", self.get(row, column));
            }
            println!("]");
        }
    }
}

// Arithmetic operations.
impl<T> Matrix<T>
where
    T: Add<Output = T> + Copy,
{
    pub fn add(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Different number of rows");
        assert_eq!(self.columns, other.columns, "Different number of columns");

        let mut data: Vec<T> = Vec::new();

        for i in 0..self.data.len() {
            let value = self.data[i] + other.data[i];
            data.push(value);
        }

        Matrix::new(self.rows, self.columns, data)
    }
}

impl<T> Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    pub fn sub(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Different number of rows");
        assert_eq!(self.columns, other.columns, "Different number of columns");

        let mut data: Vec<T> = Vec::new();

        for i in 0..self.data.len() {
            let value = self.data[i] - other.data[i];
            data.push(value);
        }

        Matrix::new(self.rows, self.columns, data)
    }
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
   
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    pub fn multiply_scalar(&self, escalar: T) -> Matrix<T> {
        let mut data: Vec<T> = Vec::new();

        for i in 0..self.data.len() {
            data.push(escalar * self.data[i]);
        }
        Matrix::new(self.rows, self.columns, data)
    }
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Add<Output = T> + Default + Copy,
{
    pub fn multiply(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.columns, other.rows,
            "Columns of the first matrix must equal rows of the second matrix"
        );

        let mut data: Vec<T> = Vec::new();

        for i in 0..self.rows {
            for k in 0..other.columns {
                let mut sum = T::default();

                for j in 0..self.columns {
                    let a = self.data[i * self.columns + j];
                    let b = other.data[j * other.columns + k];

                    sum = sum + (a * b);
                }

                data.push(sum);
            }
        }

        Matrix::new(self.rows, other.columns, data)
    }
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + Copy,
{
    pub fn add_bias(&self, bias: &Matrix<T>) -> Matrix<T> {
        assert_eq!(bias.rows, 1, "Bias must have one row");
        assert_eq!(bias.columns, self.columns, "Incompatible dimensions");
        let mut data: Vec<T> = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.columns {
                let sum = self.data[j + (self.columns * i)] + bias.data[j];
                data.push(sum);
            }
        }

        Matrix {
            rows: (self.rows),
            columns: (self.columns),
            data: (data),
        }
    }
}

impl<T> Matrix<T>
where
    T: Copy,
{
    pub fn map<F>(&self, f: F) -> Matrix<T>
    where
        F: Fn(T) -> T,
    {
        let mut data: Vec<T> = Vec::new();
        for i in 0..self.rows{
            for j in 0..self.columns{
                let value = self.data[j + (self.columns*i)];
                let result = f(value);
                data.push(result);
            }
        }

        Matrix {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }
}
impl<T> Matrix<T>
where
    T: From<u8> + Copy,
{
    pub fn zeros(rows: usize, columns: usize) -> Matrix<T> {
        let mut data: Vec<T> = Vec::new();

        for _ in 0..(rows * columns) {
            data.push(T::from(0));
        }

        Matrix::new(rows, columns, data)
    }
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    pub fn multiply_elementwise(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Different number of rows");
        assert_eq!(self.columns, other.columns, "Different number of columns");

        let mut data: Vec<T> = Vec::new();

        for i in 0..self.data.len() {
            let value = self.data[i] * other.data[i];
            data.push(value);
        }

        Matrix::new(self.rows, self.columns, data)
    }
}
use rand::Rng;

impl Matrix<f32> {
    pub fn random(rows: usize, columns: usize) -> Matrix<f32> {
        let mut rng = rand::rng();
        let mut data: Vec<f32> = Vec::new();

        for _ in 0..(rows * columns) {
            let value = rng.random_range(-1.0..1.0);
            data.push(value);
        }

        Matrix::new(rows, columns, data)
    }
}
