use rand::rngs::ThreadRng;
#[allow(unused_imports)]
use rand::thread_rng;
use rayon::prelude::*;
use std::time::Instant;
use textplots::{utils, Chart, Plot, Shape};

mod approx;
use approx::Approx;
mod minihist;
use minihist::MiniHist;

struct Population(Vec<Approx>);

impl Population {
    fn with_capacity(n: usize) -> Population {
        let approx: Vec<Approx> = (0..n)
            .map(|i| Approx::from_seed(i as u64 * 11111))
            .collect();
        // Initial population:
        // 0x5f1ffff9, 0.703952253, 2.38924456
        // 0x5f601800, 0.2485, 4.7832
        Population(approx)
    }
    fn evolve(&mut self) {
        // TODO: random walk nkeep fraction as well
        let nkeep = (0.05 * self.0.len() as f32) as usize;
        let mut t = 1;
        loop {
            let (_, to_mutate) = self.0.split_at_mut(nkeep);
            to_mutate.par_iter_mut().for_each(|c| c.mutate());
            to_mutate.par_iter_mut().for_each(|a| a.calculate_fitness());
            self.0.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // let tscale: u32 = 100 * t / nt + 1;
            // nkeep = 1 + 99 / tscale as usize;
            // let best_max_error = self.approx[0].max_error.1;
            // let thresh = 0.1 / (t as f32);
            // for (i, a) in self.approx.iter().take(50).enumerate() {
            // if a.max_error.1 > 1.01  * best_max_error {
            // nkeep = i;
            // break;
            // }
            // }
            if t % 1000 == 0 {
                // let mut hist = MiniHist::with_range(-5.0, 5.0, 50);
                // for a in self.approx.iter() {
                //     hist.add(a.max_error.1.log(10.0));
                // }
                // let hist = hist.as_tuple_vec();
                // Chart::new(180, 60, -5.0, 5.0)
                //     .lineplot(&Shape::Bars(&hist[..]))
                //     .nice();
                println!("{:8}. {} (kept {})", t, self.0[0], nkeep);
            }
            // fill rest of population with offspring of keepers
            for child in 2 * nkeep..self.0.len() {
                let p = child % nkeep;
                self.0[child].c1 = self.0[p].c1;
                self.0[child].c2 = self.0[p].c2;
                self.0[child].c3 = self.0[p].c3;
            }
            t += 1;
        }
    }
}

fn main() {
    let mut p = Population::with_capacity(512);
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
    p.evolve();
    println!("{:?} elapsed", Instant::now() - start);
}
