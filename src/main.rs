use std::cmp::Ordering;

use std::fmt::{self, Display};
use std::time::Instant;

// use rand::distributions::{Distribution, Uniform};
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;


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
    fn mutate(&mut self, t: u32, nt: u32) {
        // let old_coefs: (u32, f32, f32) = (self.c1, self.c2, self.c3);
        let r: f32 = self.rng.gen();
        let sigma: f32 = 0.001f32;
        let val: f32 = self.rng.sample(StandardNormal);
        if r < 0.3 {
            self.c1 *= 1 + (self.c1 as f32 * val * sigma) as u32;
        } else if r < 0.6 {
            self.c2 *= 1f32 + val * sigma;
        } else if r < 0.9 {
            self.c3 *= 1f32 + val * sigma;
        } else {
            let w = 20 + nt/t;
            self.c1 += self.rng.gen_range(0..w) - w/2;
            self.c2 *= 1f32 + val * sigma;
            let val: f32 = self.rng.sample(StandardNormal);
            self.c3 *= 1f32 + val * sigma;
        }
        // println!("{} {:?} -> ({},{},{})", r,    old_coefs, self.c1, self.c2, self.c3);
    }
    fn inv_sqrt(&self, x: f32) -> f32 {
        let mut y = FloatInt { f: x };
        y.u = unsafe { self.c1 - (y.u >> 1) };
        return self.c2 * unsafe { y.f * (self.c3 - x * y.f * y.f) };
    }
    fn approx_error(&mut self, x: f32) -> f32 {
        let y_approx = self.inv_sqrt(x);
        let y_true = 1.0f32 / x.sqrt();
        (y_approx - y_true).abs()
    }
    fn update_fitness(&mut self) {
        let mut top_errors = [(0f32, 0f32); 4];
        for i in 0..512 {
            let x = 1.0f32 + (i as f32) * 3.0f32 / 512f32;
            let error = self.approx_error(x);
            for j in 0..4 {
                if top_errors[j].1 < error {
                    top_errors[j].1 = error;
                    top_errors[j].0 = x;
                }
            }
        }
        self.max_error = top_errors[0].1;
    }
    fn step(&mut self, t: u32, nt: u32) {
        self.mutate(t+1, nt);
        self.update_fitness();
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
    fn evolve(&mut self, nt: u32) {
        for t in 0..nt {
            self.coefs.par_iter_mut().for_each(|c| c.step(t, nt));
            self.coefs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nkeep = 10;
            for i in nkeep..self.coefs.len()/2 {
                self.coefs[i].c1 = self.coefs[i % nkeep].c1;
                self.coefs[i].c2 = self.coefs[i % nkeep].c2;
                self.coefs[i].c3 = self.coefs[i % nkeep].c3;
            }
            if t % (nt / 100) == 0 {
                println!(
                    "[{}] best: fitness={}  {}", t,
                    self.coefs[0].max_error, self.coefs[0]
                );
            }
        }
    }
}

fn main() {
    let mut p = Population::with_capacity(100);
    let mut coefs_start = p.coefs[0].clone();
    coefs_start.update_fitness();
    let start = Instant::now();
    p.evolve(1000000);
    println!("start: {} {}\nend:   {} {}", coefs_start.max_error, coefs_start,
        p.coefs[0].max_error, p.coefs[0]);
    println!("{:?} elapsed", Instant::now() - start);
}
