use std::cmp::Ordering;
use std::fmt::{self, Display};
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use num::clamp;


#[repr(C)]
union FloatInt {
    u: u32,
    f: f32,
}

fn inv_sqrt(x: f32, c1: u32, c2: f32, c3: f32) -> f32 {
    let mut y = FloatInt { f: x };
    // self.c1 = (3/2) x 2^23 x (127 - mu) = 0x5f400000 - 0xc00000*mu,
    // where log(1+x) ~= x + mu
    // SAFETY: y is local and not borrowed
    y.u = unsafe { c1 - (y.u >> 1) };
    // self.c2, self.c3 come from Newton-Raphson iteration
    // SAFETY: y is local and not borrowed
    return c2 * unsafe { y.f * (c3 - x * y.f * y.f) };
}
fn inv_sqrt_error(x: f32, c1: u32, c2: f32, c3: f32) -> f32 {
    //assert!(x >= 0.0f32);
    //assert!(x <= 4.0f32);
    let y_approx = inv_sqrt(x, c1, c2, c3);
    let y_true = 1.0f32 / x.sqrt();
    let e = (y_approx - y_true).abs() / y_true;
    if e.is_nan() {
        100f32
    } else {
        e
    }
}

#[derive(Clone)]
pub struct Approx {
    rng: Pcg64,
    pub c1: u32,
    pub c2: f32,
    pub c3: f32,
    pub max_error: f32,
    /// x position where max error occurs
    pub max_error_loc: f32,  
    pub rms_error: f32,
}

impl Approx {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        // 0x5f601800, 0.2485, 4.7832
        //c1: 0x5f601800, c2: 0.2485, c3: 4.7832,
        let r: f32 = rng.gen();
        let (c1, c2, c3): (u32, f32, f32) = if r < 0.25 {
            // lowest max error found so far
            (0x5f5e555c, 0.255280614, 4.698304653)
        } else if r < 0.5 {
            // lowest RMS error found so far
            (0x5f1abf31, 0.759093463, 2.271862507)
        } else if r < 0.75 {
            // Kadlec's:
            (0x5F1FFFF9, 0.703952253, 2.38924456)
        } else {
            // totally random
            let c1 = rng.gen_range(0x59400000..0x5f400000);
            let c2: f32 = rng.gen();
            let c3: f32 = 3.0 * rng.gen::<f32>() + 2.0;
            (c1, c2, c3)
        };
        let mut a = Approx {
            rng,
            c1, c2, c3,
            max_error: std::f32::NAN,
            max_error_loc: 0.0,
            rms_error: std::f32::NAN,
        };
        a.search_interval();
        a
    }
    // pub fn coefs(&self) -> (u32, f32, f32) {
    //     (self.c1, self.c2, self.c3)
    // }
    fn mutate(&mut self, _t: u32, _nt: u32) {
        let r: f32 = self.rng.gen();
        let val: f32 = self.rng.sample(StandardNormal);
        if r < 0.3 {
            self.c1 = (self.c1 as i32 + (self.c1 as f32 * val * 0.01f32) as i32) as u32;
            // let w = 0xc000;
            // self.c1 = self.c1.saturating_add(self.rng.gen_range(0..w) - w/2);
            //self.c1 = self.c1.max(0x5f000000).min(0x5f600000);
        } else if r < 0.6 {
            self.c2 *= 1f32 + val;
            self.c2 = clamp(self.c2, 2.0, 5.0);
        } else if r < 0.9 {
            self.c3 *= 1f32 + val;
            self.c3 = clamp(self.c3, 0.0, 1.0);
        } else {
            //self.c1 = self.rng.gen_range(0x59400000..0x5f400000);
            //self.c2 = self.rng.gen();
            //self.c3 = 3.0 * self.rng.gen::<f32>() + 2.0;
            self.c1 = self.rng.gen_range(0x5f000000..0x5f600000);
            self.c2 *= 1f32 + val * 0.1f32;
            let val: f32 = self.rng.sample(StandardNormal);
            self.c3 *= 1f32 + val * 0.1f32;
        }
        // println!("{} {:?} -> ({},{},{})", r,    old_approx, self.c1, self.c2, self.c3);
    }
    // pub fn from_sex(&mut self, coefs1: (u32, f32, f32), coefs2: (u32, f32, f32)) {
    //     self.c1 = if self.rng.gen_bool(0.5) { coefs1.0 } else { coefs2.0 };
    //     self.c2 = if self.rng.gen_bool(0.5) { coefs1.1 } else { coefs2.1 };
    //     self.c3 = if self.rng.gen_bool(0.5) { coefs1.2 } else { coefs2.2 };
    // }

    /// Find the peak of the function `error` within the interval `(a, b)`
    fn peak_find_x(&self, a: f32, b: f32) -> (f32, f32) {
        let mut l = a;
        let mut r = b;
        loop {
            let m = 0.5*(r + l); // bisect interval
            // we know that grad(a) > 0 and grad(b) < 0, so all that matters is grad at midpoint
            let grad_mid = self.derror_dx(m);
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
        let yhat = inv_sqrt_error(xhat, self.c1, self.c2, self.c3);
        (xhat, yhat)
    }
    /// Find the peak of the function `error` within the interval `(a, b)`
    // fn peak_find_c23(&mut self, c2: [f32; 2], c3: [f32; 2]) -> (f32, f32) {
    //     let mut l = a;
    //     let mut r = b;
    //     loop {
    //         let m = 0.5*(r + l); // bisect interval
    //         // we know that grad(a) > 0 and grad(b) < 0, so all that matters is grad at midpoint
    //         let grad_mid = self.derror_dc23(m);
    //         // println!("[{}, {}] grad_mid={}", l, r, grad_mid);
    //         if r - l < 2f32 * f32::EPSILON || grad_mid == 0.0 { // FOUND IT! TODO: lower thresh?
    //             break;
    //         } else if grad_mid < 0.0 { // maximum is to the left
    //             r = m;
    //         } else if grad_mid > 0.0 { // maximum is to the right
    //             l = m;
    //         } else {
    //             panic!("How'd we get here?");
    //         }
    //     }
    //     let xhat = 0.5*(l + r);
    //     let yhat = inv_sqrt_error(xhat, self.c1, self.c2, self.c3);
    //     (xhat, yhat)
    // }
    // /// estimate the gradient w.r.t. `c2` and `c3` of `error` at location `x`
    // fn derror_dc23(&mut self, x: f32) -> f32 {
    //     let c2_1 = self.c2 - 100f32 * f32::EPSILON;
    //     let c2_2 = self.c2 + 100f32 * f32::EPSILON;
    //     let y1 = inv_sqrt_error(x, self.c1, c2_1, self.c3);
    //     let y2 = inv_sqrt_error(x, self.c1, c2_2, self.c3);
    //     let dx = 200f32 * f32::EPSILON;
    //     (y2 - y1) / dx
    // }
    /// estimate the gradient w.r.t. `x` of `error` at location `x`
    fn derror_dx(&self, x: f32) -> f32 {
        let x1 = x - f32::EPSILON;
        let x2 = x + f32::EPSILON;
        let y1 = inv_sqrt_error(x1, self.c1, self.c2, self.c3);
        let y2 = inv_sqrt_error(x2, self.c1, self.c2, self.c3);
        let dx = 2f32 * f32::EPSILON;
        (y2 - y1) / dx
    }
    /// find local optimum of c2 and c3
    //fn optimize_newton_step(&mut self) {
    //    unimplemented!()
   // }
    pub fn search_interval(&mut self) {
        const NDIV: u32 = 512;
        let mut errors: Vec<(f32, f32)> = Vec::with_capacity(NDIV as usize);
        self.rms_error = 0.0;
        // coarsely explore interval [1,4)
        for i in 0..NDIV {
            let x = 1f32 + (i as f32) * 3f32 / (NDIV as f32);
            let error = inv_sqrt_error(x, self.c1, self.c2, self.c3);
            errors.push((x, error));
            self.rms_error += error*error;
        }
        self.rms_error /= NDIV as f32;
        errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.max_error = errors[0].1;
        self.max_error_loc = errors[0].0;
        // explore regions around top-4 errors
        for i in 0..4 {
            // x is center, and interval is (x - 3/NSTEPS, x + 3/NSTEPS]
            let x1 = errors[i].0 - 3.0f32 / (NDIV as f32);
            let x2 = errors[i].0 + 3.0f32 / (NDIV as f32);
            if self.derror_dx(x1) > 0.0 && self.derror_dx(x2) < 0.0 { // we bracket a peak
                // now bisect to find the max
                let res = self.peak_find_x(x1, x2);
                // println!("errors[{}] = {:?} => {:?}", i, errors[i], res);
                errors[i] = res;
                if self.max_error < errors[i].1 {
                    self.max_error = errors[i].1;
                    self.max_error_loc = errors[i].0;
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
            c1: 0x5f3759df,
            c2: 1.5,
            c3: 0.5,
            max_error: std::f32::MAX,
            max_error_loc: 0.0,
            rms_error: std::f32::MAX,
        };
        a.search_interval();
        a
    }
}

impl PartialOrd for Approx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let me = (self.rms_error, self.max_error);
        let them = (other.rms_error, other.max_error);
        //self.max_error.1.partial_cmp(&other.max_error.1)
        //self.rms_error.partial_cmp(&other.rms_error)
        me.partial_cmp(&them)
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
            "{:.10},{},{:e},{:x},{:.9},{:1.9}",
            self.max_error, self.max_error_loc, self.rms_error, self.c1, self.c2, self.c3
        )
    }
}
