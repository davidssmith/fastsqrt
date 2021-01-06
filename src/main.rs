use std::fmt::{self, Display};

#[repr(C)]
union MyUnion {
    u: u32,
    f: f32,
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Coefs {
    c1: u32,
    c2: f32,
    c3: f32,
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

fn inv_sqrt(x: f32, c: Coefs) -> f32 {
    let mut y = MyUnion { f: x };
    y.u = unsafe { c.c1 - (y.u >> 1)};
    return c.c2 * unsafe { y.f * (c.c3 - x * y.f * y.f)};
}

fn main() {
    let c = Coefs::default();
    println!("{}", c);
    let x = 3.0f32;
    println!("isqrt({}) = {}", x, inv_sqrt(x, c));
}
