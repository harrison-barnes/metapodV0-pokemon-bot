//! Randomness abstraction. The battle engine asks a `Chance` for every random event, so the same
//! code can either sample one battle (`Sampler`) or enumerate every outcome exactly (`Enumerator`).

pub trait Chance {
    /// Picks an index with probability `weights[i] / sum(weights)`. Zero-weight entries are never picked.
    fn pick(&mut self, weights: &[u32]) -> usize;

    /// `pick` for weights that are all positive and whose sum is already known.
    fn pick_positive(&mut self, weights: &[u32], _total: u64) -> usize {
        self.pick(weights)
    }

    /// `call BattleRandom; cp threshold; jr c` — true with probability threshold/256.
    fn below(&mut self, threshold: u32) -> bool {
        if threshold == 0 {
            return false;
        }
        if threshold >= 256 {
            return true;
        }
        self.pick(&[threshold, 256 - threshold]) == 0
    }
}

/// Small, fast PRNG (xoshiro256**) so sampling needs no external crate.
#[derive(Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in 0..n.
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub struct Sampler {
    pub rng: Rng,
}

impl Chance for Sampler {
    fn pick(&mut self, weights: &[u32]) -> usize {
        let total: u64 = weights.iter().map(|&w| w as u64).sum();
        let mut r = self.rng.below(total);
        for (i, &w) in weights.iter().enumerate() {
            if r < w as u64 {
                return i;
            }
            r -= w as u64;
        }
        unreachable!()
    }
}

/// Enumerates every execution path of a deterministic-given-choices program.
///
/// Usage: `loop { run(&mut e); record(e.prob()); if !e.advance() { break } }`. Each run replays the
/// recorded prefix of choices and extends it with first choices; `advance` moves to the next path.
#[derive(Default)]
pub struct Enumerator {
    path: Vec<u16>,
    sizes: Vec<u16>,
    depth: usize,
    prob: f64,
}

impl Enumerator {
    pub fn new() -> Enumerator {
        Enumerator { prob: 1.0, ..Default::default() }
    }

    pub fn prob(&self) -> f64 {
        self.prob
    }

    /// Moves to the next unexplored path. Returns false when all paths have been visited.
    pub fn advance(&mut self) -> bool {
        // Choices recorded beyond the depth reached by the last run belong to no path.
        self.path.truncate(self.depth);
        self.sizes.truncate(self.depth);
        while let Some(last) = self.path.last_mut() {
            *last += 1;
            if *last < *self.sizes.last().unwrap() {
                self.depth = 0;
                self.prob = 1.0;
                return true;
            }
            self.path.pop();
            self.sizes.pop();
        }
        false
    }
}

impl Enumerator {
    #[inline]
    fn next_choice(&mut self, n: u16) -> u16 {
        let k = if self.depth < self.path.len() {
            self.path[self.depth]
        } else {
            self.path.push(0);
            self.sizes.push(n);
            0
        };
        self.depth += 1;
        k
    }
}

impl Chance for Enumerator {
    fn pick_positive(&mut self, weights: &[u32], total: u64) -> usize {
        let k = self.next_choice(weights.len() as u16) as usize;
        self.prob *= weights[k] as f64 / total as f64;
        k
    }

    fn below(&mut self, threshold: u32) -> bool {
        if threshold == 0 {
            return false;
        }
        if threshold >= 256 {
            return true;
        }
        let k = self.next_choice(2);
        if k == 0 {
            self.prob *= threshold as f64 / 256.0;
        } else {
            self.prob *= (256 - threshold) as f64 / 256.0;
        }
        k == 0
    }

    fn pick(&mut self, weights: &[u32]) -> usize {
        let mut total: u64 = 0;
        let mut nonzero = 0u16;
        for &w in weights {
            total += w as u64;
            nonzero += (w > 0) as u16;
        }
        let k = if self.depth < self.path.len() {
            self.path[self.depth]
        } else {
            self.path.push(0);
            self.sizes.push(nonzero);
            0
        };
        self.depth += 1;
        // k-th nonzero entry
        let mut seen = 0u16;
        let mut idx = 0;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0 {
                if seen == k {
                    idx = i;
                    break;
                }
                seen += 1;
            }
        }
        self.prob *= weights[idx] as f64 / total as f64;
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerates_all_paths_with_total_probability_one() {
        let mut e = Enumerator::new();
        let mut total = 0.0;
        let mut n = 0;
        loop {
            // A variable-depth program: second draw only happens on the first branch.
            let a = e.pick(&[1, 2, 0, 1]);
            if a == 0 {
                e.below(64);
            }
            total += e.prob();
            n += 1;
            if !e.advance() {
                break;
            }
        }
        assert_eq!(n, 4);
        assert!((total - 1.0).abs() < 1e-12);
    }
}
