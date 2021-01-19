
use std::time::Instant;
use textplots::{utils, Chart, Plot, Shape};
use rayon::prelude::*;

mod approx;
use approx::Approx;
mod minihist;
use minihist::MiniHist;

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
        let mut nkeep = 50;
        for t in 1..nt {
            self.approx[nkeep..].par_iter_mut().for_each(|c| c.step(t, nt));
            self.approx.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let tscale: u32 = 100 * t / nt + 1;
            nkeep = 1 + 49 / tscale as usize;
            // let best_max_error = self.approx[0].max_error.1;
            // let thresh = 0.1 / (t as f32);
            // for (i, a) in self.approx.iter().take(50).enumerate() {
                // if a.max_error.1 > 1.01  * best_max_error {
                    // nkeep = i;
                    // break;
                // }
            // }
            if t % 1000 == 0 {
                let mut hist = MiniHist::with_range(4.0, 8.0, 20);
                for a in self.approx.iter() {
                    hist.add(a.max_error.1*10000f32);
                }
                let hist = hist.as_tuple_vec();
                Chart::new(180, 60, 4.0, 8.0)
                    .lineplot(&Shape::Bars(&hist[..]))
                    .nice();
                println!("{:8}. {} (kept {})", t, self.approx[0], nkeep);
            }
            // fill rest of population with offspring of keepers
            for child in 2*nkeep..self.approx.len() {
                let p = child % nkeep;
                self.approx[child].c1 = self.approx[p].c1;
                self.approx[child].c2 = self.approx[p].c2;
                self.approx[child].c3 = self.approx[p].c3;
            }


        }
    }
}

fn main() {
    let mut p = Population::with_capacity(1000);
    let mut approx_start = p.approx[0].clone();
    approx_start.search_interval();
    let start = Instant::now();
    // println!(
    //     "{}",
    //     plot(
    //         vec![
    //             0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -0.5, 9.0, -3.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0,
    //             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -0.5, 8.0, -3.0, 0.0, 0.0, 1.0,
    //             2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -0.5, 10.0, -3.0,
    //             0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0
    //         ],
    //         Config::default()
    //             .with_offset(10)
    //             .with_height(10)
    //             .with_caption("I'm a doctor, not an engineer.".to_string())
    //     )
    // );
    p.evolve(100_000_000);
    println!("start: {}\nend:   {}", approx_start, p.approx[0]);
    println!("{:?} elapsed", Instant::now() - start);
}
