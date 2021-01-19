
use std::time::Instant;
use rayon::prelude::*;

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
            for child in 2*nkeep..self.approx.len() {
                let p = child % nkeep;
                self.approx[child].c1 = self.approx[p].c1;
                self.approx[child].c2 = self.approx[p].c2;
                self.approx[child].c3 = self.approx[p].c3;
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
