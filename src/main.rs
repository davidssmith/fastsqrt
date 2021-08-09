#![warn(dead_code)]

use std::time::Instant;
use rayon::prelude::*;

mod approx;
use approx::Approx;
//mod minihist;
//use minihist::MiniHist;

struct Population {
    rms_approx: Vec<Approx>,
    minmax_approx: Vec<Approx>,
}

impl Population {
    fn with_capacity(n: usize) -> Population {
        let rms_approx: Vec<Approx> = (0..n)
            .map(|i| Approx::from_seed(0x1337 + 0xc0ffee * i as u64))
            .collect();
        let minmax_approx: Vec<Approx> = (0..n)
            .map(|i| Approx::from_seed(0x1337 + 0xc0ffee * i as u64))
            .collect();
        // Initial population:
        // 0x5f1ffff9, 0.703952253, 2.38924456
        // 0x5f601800, 0.2485, 4.7832
        Population { rms_approx, minmax_approx }
    }
    fn evolve(&mut self, nt: u32) {
        let nkeep = self.rms_approx.len().div_euclid(10) as usize; 
        let mut t = 1;
        println!("goal,gen,max,maxloc,rms,c1,c2,c3,nkeep,tper(ns)");
        println!("rms,{},{},{},", t, self.rms_approx[0], nkeep);
        println!("max,{},{},{},", t, self.minmax_approx[0], nkeep);
        let start = Instant::now();
        loop {
            self.rms_approx[nkeep..].par_iter_mut().for_each(|c| c.step(t, nt));
            self.rms_approx.sort_by(|a,b| a.rms_error.partial_cmp(&b.rms_error).unwrap());
            self.minmax_approx[nkeep..].par_iter_mut().for_each(|c| c.step(t, nt));
            self.minmax_approx.sort_by(|a,b| a.max_error.partial_cmp(&b.max_error).unwrap());
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
                //let mut hist = MiniHist::with_range(-5.0, 3.0, 64);
                //for a in self.approx.iter() {
                //    hist.add(a.max_error.1.log(10.0));
                //}
                //let hist = hist.as_tuple_vec();
                //Chart::new(180, 60, -5.0, 3.0)
                //    .lineplot(&Shape::Bars(&hist[..]))
                //    .nice();
                let tper = start.elapsed().as_nanos() / (t as u128) / (self.rms_approx.len() as u128);
                println!("rms,{},{},{},{}", t, self.rms_approx[0], nkeep, tper);
                let tper = start.elapsed().as_nanos() / (t as u128) / (self.minmax_approx.len() as u128);
                println!("max,{},{},{},{}", t, self.minmax_approx[0], nkeep, tper);
            }
            // fill rest of population with offspring of keepers
            for child in 2*nkeep..self.rms_approx.len() {
                let p = child % nkeep;
                self.rms_approx[child].c1 = self.rms_approx[p].c1;
                self.rms_approx[child].c2 = self.rms_approx[p].c2;
                self.rms_approx[child].c3 = self.rms_approx[p].c3;
            }
            for child in 2*nkeep..self.minmax_approx.len() {
                let p = child % nkeep;
                self.minmax_approx[child].c1 = self.minmax_approx[p].c1;
                self.minmax_approx[child].c2 = self.minmax_approx[p].c2;
                self.minmax_approx[child].c3 = self.minmax_approx[p].c3;
            }
            t += 1;
        }
    }
}

fn main() {
    let mut p = Population::with_capacity(10000);
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
}
