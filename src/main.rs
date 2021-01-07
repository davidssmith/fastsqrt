use std::fmt::{self, Display};
// use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_distr::StandardNormal;

#[repr(C)]
union FloatInt {
    u: u32,
    f: f32,
}

struct Fitness {
    max: f32,
    rms: f32,
}


#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Coefs {
    c1: u32,
    c2: f32,
    c3: f32,
}

impl Coefs {
    fn mutate(&mut self) {
        let r: f32 = thread_rng().gen();
        let val: f32 = thread_rng().sample(StandardNormal);
        if r < 0.16 {
            self.c1 += 1;
        } else if r < 0.32 {
            self.c1 -= 1;
        } else if r < 0.67 {
            self.c2 = self.c2 * (1f32 + val*0.01f32);
        } else {
            self.c3 = self.c3 * (1f32 + val*0.01f32);
        }
    }
    fn inv_sqrt(&self, x: f32) -> f32 {
        let mut y = FloatInt { f: x };
        y.u = unsafe { self.c1 - (y.u >> 1)};
        return self.c2 * unsafe { y.f * (self.c3 - x * y.f * y.f)};
    }
    fn fitness(&self) -> f32 {
        // let mut errors = [0f32; 512];
        let mut max_error: f32 = 0.0;
        for i in 0..512 {
            let x = 1.0f32 + (i as f32) * 3.0f32 / 512f32;
            let y_approx = self.inv_sqrt(x);
            let y_true = 1.0f32 / x.sqrt();
            let error = (y_approx - y_true).abs();
            max_error = max_error.max(error);
        }
        max_error
    }
}

impl Default for Coefs {
    fn default() -> Self {
        Coefs { c1: 0x5f1ffff9u32, c2: 0.703952253f32, c3: 2.38924456f32 }
    }
}

impl Display for Coefs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y.u = 0x{:x} - (y.u >> 1); {:.9}f * y.f * ({:1.9}f - x * y.f * y.f)",
            self.c1, self.c2, self.c3)
    }
}

struct Population {
    p: Vec<Coefs>,
}

impl Population {
    fn with_capacity(n: usize) -> Population {
        let mut p = Vec::with_capacity(n);
        for _ in 0..n {
            p.push(Coefs::default());
        }
        Population { p }
    }
    fn len(&self) -> usize {
        self.p.len()
    }
    fn mutate(&mut self) {
        self.p.iter_mut().for_each(|p| p.mutate());
    }
    fn evolve(&mut self) {

    }
    fn best_fitness(&self) -> f32 {
        let fitnesses = self.p.iter().map(|p| p.fitness()).collect::<Vec<f32>>();
        let mut min_fitness = 1.0f32;
        for f in fitnesses.iter() {
            min_fitness = min_fitness.min(*f);
        }
        min_fitness
    }
}

fn main() {
    // let mut rng = rand::thread_rng();
    let mut p = Population::with_capacity(100);
    p.mutate();
    println!("fitness = {}", p.best_fitness());
}
