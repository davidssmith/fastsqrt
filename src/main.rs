use std::cmp::Ordering;

use std::fmt::{self, Display};

// use rand::distributions::{Distribution, Uniform};
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;

#[repr(C)]
union FloatInt {
    u: u32,
    f: f32,
}

#[derive(Clone)]
struct Coefs {
    rng: Pcg64,
    c1: u32,
    c2: f32,
    c3: f32,
    max_error: f32,
    rms_error: f32,
}

impl Coefs {
    fn from_seed(seed: u64) -> Self {
        Coefs {
            rng: Pcg64::seed_from_u64(seed),
            c1: 0x5f1ffff9u32,
            c2: 0.703952253f32,
            c3: 2.38924456f32,
            max_error: std::f32::MAX,
            rms_error: std::f32::MAX,
        }
    }
    fn mutate(&mut self) {
        // let old_coefs: (u32, f32, f32) = (self.c1, self.c2, self.c3);
        let r: f32 = self.rng.gen();
        if r < 0.3 {
            self.c1 += self.rng.gen_range(0..21) - 10;
        } else if r < 0.6 {
            let val: f32 = self.rng.sample(StandardNormal);
            self.c2 = self.c2 * (1f32 + val * 0.01f32);
        } else if r < 0.9 {
            let val: f32 = self.rng.sample(StandardNormal);
            self.c3 = self.c3 * (1f32 + val * 0.01f32);
        } else {
            self.c1 += self.rng.gen_range(0..21) - 10;
            let val: f32 = self.rng.sample(StandardNormal);
            self.c2 = self.c2 * (1f32 + val * 0.01f32);
            let val: f32 = self.rng.sample(StandardNormal);
            self.c3 = self.c3 * (1f32 + val * 0.01f32);
        }
        // println!("{} {:?} -> ({},{},{})", r,    old_coefs, self.c1, self.c2, self.c3);
    }
    fn inv_sqrt(&self, x: f32) -> f32 {
        let mut y = FloatInt { f: x };
        y.u = unsafe { self.c1 - (y.u >> 1) };
        return self.c2 * unsafe { y.f * (self.c3 - x * y.f * y.f) };
    }
    fn fitness(&mut self) {
        // let mut errors = [0f32; 512];
        self.max_error = 0.0;
        for i in 0..2048 {
            let x = 1.0f32 + (i as f32) * 3.0f32 / 2048f32;
            let y_approx = self.inv_sqrt(x);
            let y_true = 1.0f32 / x.sqrt();
            let error = (y_approx - y_true).abs();
            self.max_error = self.max_error.max(error);
        }
    }
}

impl Default for Coefs {
    fn default() -> Self {
        Coefs {
            rng: Pcg64::from_rng(thread_rng()).unwrap(),
            c1: 0x5f1ffff9u32,
            c2: 0.703952253f32,
            c3: 2.38924456f32,
            max_error: std::f32::MAX,
            rms_error: std::f32::MAX,
        }
    }
}

impl PartialOrd for Coefs {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.max_error.partial_cmp(&other.max_error)
    }
}

// impl Ord for Coefs {
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.max_error.cmp(&other.max_error)
//     }
// }

impl PartialEq for Coefs {
    fn eq(&self, other: &Self) -> bool {
        self.c1 == other.c1 && self.c2 == other.c2 && self.c3 == other.c3
    }
}

impl Eq for Coefs {}

impl Display for Coefs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "y.u = 0x{:x} - (y.u >> 1); {:.9}f * y.f * ({:1.9}f - x * y.f * y.f)",
            self.c1, self.c2, self.c3
        )
    }
}

struct Population {
    coefs: Vec<Coefs>,
}

impl Population {
    fn with_capacity(n: usize) -> Population {
        let coefs: Vec<Coefs> = (0..n)
            .map(|i| Coefs::from_seed(0x1337 + 0xc0ffee * i as u64))
            .collect();
        Population { coefs }
    }
    // fn len(&self) -> usize {
    //     self.coefs.len()
    // }
    fn mutate(&mut self) {
        self.coefs[4..].iter_mut().for_each(|c| c.mutate());
    }
    fn evolve(&mut self, nt: u32) {
        for t in 0..nt {
            self.mutate();
            self.fitnesses();
            self.coefs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nkeep = 2;
            for i in nkeep..self.coefs.len() {
                self.coefs[i].c1 = self.coefs[i % nkeep].c1;
                self.coefs[i].c2 = self.coefs[i % nkeep].c2;
                self.coefs[i].c3 = self.coefs[i % nkeep].c3;
            }
            if t % (nt / 100) == 0 {
                println!(
                    "best: fitness={}  {}",
                    self.coefs[0].max_error, self.coefs[0]
                );
            }
        }
    }
    fn fitnesses(&mut self) {
        self.coefs.iter_mut().for_each(|p| p.fitness());
    }
    // fn best_fitness(&self) -> f32 {
    //     self.coefs[0].max_error
    // }
}

fn main() {
    let mut p = Population::with_capacity(1000);
    let mut coefs_start = p.coefs[0].clone();
    coefs_start.fitness();
    p.evolve(100000);
    println!("start: {} {}\nend:   {} {}", coefs_start.max_error, coefs_start,
        p.coefs[0].max_error, p.coefs[0])
}
