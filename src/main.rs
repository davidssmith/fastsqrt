
use std::time::Instant;
use rayon::prelude::*;
use itertools::Itertools;

mod approx;
use approx::Approx;

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
            let nkeep = 20;
            self.approx[nkeep..].par_iter_mut().for_each(|c| c.step(t, nt));
            self.approx.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // fill rest of population with offspring of keepers
            let mut child = nkeep;
            for p in (1..nkeep).combinations(2) {
                let c1 = self.approx[p[0]].coefs();
                let c2 = self.approx[p[1]].coefs();
                self.approx[child].from_sex(c1, c2);
                child += 1;
                if child >= self.approx.len() {
                    break;
                }
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
    p.evolve(1000000);
    println!("start: {}\nend:   {}", approx_start, p.approx[0]);
    println!("{:?} elapsed", Instant::now() - start);
}
