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

// #[derive(Clone, Copy)]
// struct ApproxError {
//     x: f32,
//     dy: f32,
// }

#[derive(Clone)]
struct Approx {
    rng: Pcg64,
    c1: u32,
    c2: f32,
    c3: f32,
    max_error: f32,
    rms_error: f32,
}

impl Approx {
    fn from_seed(seed: u64) -> Self {
        let mut a = Approx {
            rng: Pcg64::seed_from_u64(seed),
            c1: 0x5f1ffff9u32,
            c2: 0.703952253f32,
            c3: 2.38924456f32,
            max_error: std::f32::NAN,
            rms_error: std::f32::NAN,
        };
        a.search_interval();
        a
    }
    fn mutate(&mut self, t: u32, nt: u32) {
        // let old_approx: (u32, f32, f32) = (self.c1, self.c2, self.c3);
        let r: f32 = self.rng.gen();
        let val: f32 = self.rng.sample(StandardNormal);
        if r < 0.3 {
            self.c1 = (self.c1 as i32 + (self.c1 as f32 * val * 0.001f32) as i32) as u32;
        } else if r < 0.6 {
            self.c2 *= 1f32 + val * 0.01f32;
        } else if r < 0.9 {
            self.c3 *= 1f32 + val * 0.01f32;
        } else {
            let w = 20 + nt/t;
            self.c1 += self.rng.gen_range(0..w) - w/2;
            self.c2 *= 1f32 + val * 0.01f32;
            let val: f32 = self.rng.sample(StandardNormal);
            self.c3 *= 1f32 + val * 0.01f32;
        }
        // println!("{} {:?} -> ({},{},{})", r,    old_approx, self.c1, self.c2, self.c3);
    }
    fn inv_sqrt(&self, x: f32) -> f32 {
        let mut y = FloatInt { f: x };
        y.u = unsafe { self.c1 - (y.u >> 1) };
        return self.c2 * unsafe { y.f * (self.c3 - x * y.f * y.f) };
    }
    fn error(&mut self, x: f32) -> f32 {
        let y_approx = self.inv_sqrt(x);
        let y_true = 1.0f32 / x.sqrt();
        (y_approx - y_true).abs()
    }
    /// Find the peak of the function `error` within the interval `(a, b)`
    fn peak_find(&mut self, a: f32, b: f32) -> (f32, f32) {
        let mut l = a;
        let mut r = b;
        loop {
            let m = 0.5*(r + l); // bisect interval
            // we know that grad(a) > 0 and grad(b) < 0, so all that matters is grad at midpoint
            let grad_mid = self.error_gradient(m);
            // println!("[{}, {}] grad_mid={}", l, r, grad_mid);
            if r - l < 2f32 * f32::EPSILON || grad_mid == 0.0 { // FOUND IT! TODO: lower thresh?
                break;
            } else if grad_mid < 0.0 { // maximum is to the left
                r = m;
            } else if grad_mid > 0.0 { // maximum is to the right
                l = m;
            } else {
                panic!("How'd we get here?");
            }
        }
        let xhat = 0.5*(l + r);
        let yhat = self.error(xhat);
        (xhat, yhat)
    }
    /// estimate the gradient of `error` at location `x`
    fn error_gradient(&mut self, x: f32) -> f32 {
        let x1 = x - f32::EPSILON;
        let x2 = x + f32::EPSILON;
        let y1 = self.error(x1);
        let y2 = self.error(x2);
        let dx = 2f32 * f32::EPSILON;
        (y2 - y1) / dx
    }
    fn search_interval(&mut self) {
        const NDIV: u32 = 512;
        let mut errors: Vec<(f32, f32)> = Vec::with_capacity(NDIV as usize);
        self.rms_error = 0.0;
        // coarsely explore interval [1,4)
        for i in 0..NDIV {
            let x = 1f32 + (i as f32) * 3f32 / (NDIV as f32);
            let error = self.error(x);
            errors.push((x, error));
            self.rms_error += error*error;
        }
        self.rms_error /= NDIV as f32;
        errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.max_error = errors[0].1;
        // explore regions around top-4 errors
        for i in 0..4 {
            // x is center, and interval is (x - 3/NSTEPS, x + 3/NSTEPS]
            let x1 = errors[i].0 - 3.0f32 / (NDIV as f32);
            let x2 = errors[i].0 + 3.0f32 / (NDIV as f32);
            if self.error_gradient(x1) > 0.0 && self.error_gradient(x2) < 0.0 { // we bracket a peak
                // now bisect to find the max
                let res = self.peak_find(x1, x2);
                // println!("errors[{}] = {:?} => {:?}", i, errors[i], res);
                errors[i] = res;
                if self.max_error < errors[i].1 {
                    self.max_error = errors[i].1;
                }
            }
        }
    }

    fn step(&mut self, t: u32, nt: u32) {
        self.mutate(t + 1, nt);
        self.search_interval();
    }
}

impl Default for Approx {
    fn default() -> Self {
        let mut a = Approx {
            rng: Pcg64::from_rng(thread_rng()).unwrap(),
            c1: 0x5f1ffff9u32,
            c2: 0.703952253f32,
            c3: 2.38924456f32,
            max_error: std::f32::MAX,
            rms_error: std::f32::MAX,
        };
        a.search_interval();
        a
    }
}

impl PartialOrd for Approx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.max_error.partial_cmp(&other.max_error)
    }
}

impl PartialEq for Approx {
    fn eq(&self, other: &Self) -> bool {
        self.c1 == other.c1 && self.c2 == other.c2 && self.c3 == other.c3
    }
}

impl Eq for Approx {}

impl Display for Approx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max={:.10} rms={:e} y.u = 0x{:x} - (y.u >> 1); {:.9}f * y.f * ({:1.9}f - x * y.f * y.f)",
            self.max_error, self.rms_error, self.c1, self.c2, self.c3
        )
    }
}

struct Population {
    approx: Vec<Approx>,
}

impl Population {
    fn with_capacity(n: usize) -> Population {
        let approx: Vec<Approx> = (0..n)
            .map(|i| Approx::from_seed(0x1337 + 0xc0ffee * i as u64))
            .collect();
        // Initial population:
        // 0x5f1ffff9, 0.703952253, 2.38924456
        // 0x5f601800, 0.2485, 4.7832
        Population { approx }
    }
    fn evolve(&mut self, nt: u32) {
        for t in 0..nt {
            self.approx[25..].par_iter_mut().for_each(|c| c.step(t, nt));
            self.approx.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nkeep = 50;
            for i in nkeep..self.approx.len()/2 {
                self.approx[i].c1 = self.approx[i % nkeep].c1;
                self.approx[i].c2 = self.approx[i % nkeep].c2;
                self.approx[i].c3 = self.approx[i % nkeep].c3;
            }
            if t % (nt / 1000) == 0 {
                println!("{:6}. {}", t, self.approx[0]);
            }
        }
    }
}

fn main() {
    let mut p = Population::with_capacity(1000);
    let mut approx_start = p.approx[0].clone();
    approx_start.search_interval();
    let start = Instant::now();
    p.evolve(100000);
    println!("start: {} {}\nend:   {} {}", approx_start.max_error, approx_start,
        p.approx[0].max_error, p.approx[0]);
    println!("{:?} elapsed", Instant::now() - start);
}
