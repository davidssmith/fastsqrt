use itertools::Itertools;

#[derive(Clone, Default)]
pub struct MiniHist {
    min: f32,
    max: f32,
    bins: Vec<f32>,
    counts: Vec<usize>,
}

impl MiniHist {
    pub fn with_range(min: f32, max: f32, nbins: usize) -> MiniHist {
        let mut bins: Vec<f32> = Vec::with_capacity(nbins);
        for i in 0..nbins {
            bins.push((max - min) * (i as f32) / (nbins as f32) + min);
        }
        let counts = vec![0usize; nbins];
        MiniHist { min, max, bins, counts }
    }
    pub fn add(&mut self, x: f32) {
        if x < self.bins[0] {
            self.counts[0] += 1;
            return;
        }
        for i in 1..self.bins.len() {
            if self.bins[i] > x {
                self.counts[i-1] += 1;
                return;
            } else if i == self.bins.len() - 1 {
                self.counts[i] += 1;
            }
        }
    }
    pub fn as_tuple_vec(&self) -> Vec<(f32, f32)> {
        self.bins.iter().zip(self.counts.iter()).map(|(a,b)| (*a, *b as f32)).collect()
    }
}
