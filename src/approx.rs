use std::cmp::Ordering;
use std::fmt::{self, Display};
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;


#[repr(C)]
union FloatInt {
    u: u32,
    f: f32,
}


#[derive(Clone)]
pub struct Approx {
    rng: Pcg64,
    pub c1: u32,
    pub c2: f32,
    pub c3: f32,
    pub max_error: (f32, f32),
    pub rms_error: f32,
}

impl Approx {
    pub fn from_seed(seed: u64) -> Self {
        let mut a = Approx {
            rng: Pcg64::seed_from_u64(seed),
            // 0x5f601800, 0.2485, 4.7832
            c1: 0x5f1ffff9u32, c2: 0.703952253, c3: 2.38924456,
            //c1: 0x5f601800, c2: 0.2485, c3: 4.7832,
            max_error: (0f32, std::f32::NAN),
            rms_error: std::f32::NAN,
        };
        // a.c1 = a.rng.gen_range(0x5f000000..0x5fb00000);
        // a.c2 = a.rng.gen();
        // a.c3 = a.rng.gen();
        println!("{:x}, {}, {}", a.c1, a.c2, a.c3);
        a.search_interval();
        a
    }
    fn mutate(&mut self, _t: u32, _nt: u32) {
        let r: f32 = self.rng.gen();
        let val: f32 = self.rng.sample(StandardNormal);
        if r < 0.2 {
            //self.c1 = (self.c1 as i32 + (self.c1 as f32 * val * 0.001f32) as i32) as u32;
            let w = 20;
            self.c1 += self.rng.gen_range(0..w) - w/2;
        } else if r < 0.4 {
            self.c2 *= 1f32 + val * 0.01f32;
        } else if r < 0.6 {
            self.c3 *= 1f32 + val * 0.01f32;
        } else if r < 0.7 {
            self.c1 = self.rng.gen_range(0x5f000000..0x5fb00000);
        } else if r < 0.8 {
            self.c2 *= 1f32 + val * 0.5f32;
        } else if r < 0.9 {
            self.c3 *= 1f32 + val * 0.5f32;
        } else {
            self.c1 = self.rng.gen_range(0x5f000000..0x5fb00000);
            self.c2 *= 1f32 + val * 0.5f32;
            let val: f32 = self.rng.sample(StandardNormal);
            self.c3 *= 1f32 + val * 0.5f32;
        }
        // println!("{} {:?} -> ({},{},{})", r,    old_approx, self.c1, self.c2, self.c3);
    }
    fn inv_sqrt(&self, x: f32) -> f32 {
        let mut y = FloatInt { f: x };
        // self.c1 = (3/2) x 2^23 x (127 - mu) = 0x5f400000 - 0xc00000*mu,
        // where log(1+x) ~= x + mu
        y.u = unsafe { self.c1 - (y.u >> 1) };
        // self.c2, self.c3 come from Newton-Raphson iteration
        return self.c2 * unsafe { y.f * (self.c3 - x * y.f * y.f) };
    }
    fn error(&mut self, x: f32) -> f32 {
        let y_approx = self.inv_sqrt(x);
        let y_true = 1.0f32 / x.sqrt();
        (y_approx - y_true).abs() / y_true
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
    pub fn search_interval(&mut self) {
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
        self.max_error = errors[0];
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
                if self.max_error.1 < errors[i].1 {
                    self.max_error = errors[i];
                }
            }
        }
    }

    pub fn step(&mut self, t: u32, nt: u32) {
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
            max_error: (0f32, std::f32::MAX),
            rms_error: std::f32::MAX,
        };
        a.search_interval();
        a
    }
}

impl PartialOrd for Approx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.max_error.1.partial_cmp(&other.max_error.1)
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
            "max={:.10} @ {}  rms={:e}  y.u=0x{:x}-(y.u>>1); {:.9}*y.f*({:1.9}-x*y.f*y.f)",
            self.max_error.1, self.max_error.0, self.rms_error, self.c1, self.c2, self.c3
        )
    }
}
