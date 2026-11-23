//! Exact solution of small zero-sum matrix games (the simultaneous move choice in one turn).

/// Solution of a zero-sum game where the row player maximizes.
#[derive(Debug, Clone)]
pub struct GameSolution {
    pub value: f64,
    pub row: Vec<f64>,
    pub col: Vec<f64>,
}

/// Value only, with a fast path for games that have a pure saddle point.
pub fn value(a: &[f64], rows: usize, cols: usize) -> f64 {
    if rows == 1 {
        return a.iter().cloned().fold(f64::INFINITY, f64::min);
    }
    if cols == 1 {
        return a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }
    let maximin = (0..rows).map(|i| (0..cols).map(|j| a[i * cols + j]).fold(f64::INFINITY, f64::min)).fold(f64::NEG_INFINITY, f64::max);
    let minimax = (0..cols).map(|j| (0..rows).map(|i| a[i * cols + j]).fold(f64::NEG_INFINITY, f64::max)).fold(f64::INFINITY, f64::min);
    if minimax - maximin <= 1e-15 {
        return maximin;
    }
    let Tableau { t, w, shift, .. } = simplex(a, rows, cols);
    1.0 / t[rows * w + w - 1] - shift
}

/// Largest matrix game a turn can produce: at most 4 moves a side, with room to spare.
pub const MAX_CHOICES: usize = 5;
const MAX_TABLEAU: usize = (MAX_CHOICES + 1) * (2 * MAX_CHOICES + 1);

/// Simplex tableau on the stack, so solving a turn never allocates.
struct Tableau {
    t: [f64; MAX_TABLEAU],
    basis: [usize; MAX_CHOICES],
    w: usize,
    shift: f64,
}

/// With payoffs shifted to be positive, the column player's problem is
/// `max sum(y)  s.t.  A y <= 1, y >= 0`; then value = 1/sum(y), q = y * value, and the row strategy
/// comes from the dual prices of the constraints.
fn simplex(a: &[f64], rows: usize, cols: usize) -> Tableau {
    assert!(rows <= MAX_CHOICES && cols <= MAX_CHOICES, "matrix game too large: {rows}x{cols}");
    assert!(a.iter().all(|x| x.is_finite()), "non-finite payoff in matrix game: {a:?}");
    let shift = 1.0 - a.iter().cloned().fold(f64::INFINITY, f64::min);
    // Tableau: rows constraints, columns: cols decision vars + rows slacks + rhs.
    let w = cols + rows + 1;
    let mut t = [0.0f64; MAX_TABLEAU];
    for i in 0..rows {
        for j in 0..cols {
            t[i * w + j] = a[i * cols + j] + shift;
        }
        t[i * w + cols + i] = 1.0;
        t[i * w + w - 1] = 1.0;
    }
    let obj = rows * w;
    for j in 0..cols {
        t[obj + j] = -1.0;
    }
    let mut basis = [0usize; MAX_CHOICES];
    for (i, b) in basis.iter_mut().enumerate().take(rows) {
        *b = cols + i;
    }
    for _ in 0..1000 {
        // Bland's rule: lowest-index column with negative reduced cost.
        let Some(pc) = (0..cols + rows).find(|&j| t[obj + j] < -1e-12) else { break };
        let mut pr = None;
        let mut best = f64::INFINITY;
        for i in 0..rows {
            let v = t[i * w + pc];
            if v > 1e-12 {
                let ratio = t[i * w + w - 1] / v;
                if ratio < best - 1e-12 || (ratio < best + 1e-12 && pr.is_some_and(|p: usize| basis[i] < basis[p])) {
                    best = ratio;
                    pr = Some(i);
                }
            }
        }
        // Unboundedness is impossible with positive payoffs; a missing pivot row means only
        // round-off is left, so stop with the current (optimal up to round-off) basis.
        let Some(pr) = pr else { break };
        let pv = t[pr * w + pc];
        for j in 0..w {
            t[pr * w + j] /= pv;
        }
        for i in 0..=rows {
            if i == pr {
                continue;
            }
            let f = t[i * w + pc];
            if f != 0.0 {
                for j in 0..w {
                    t[i * w + j] -= f * t[pr * w + j];
                }
            }
        }
        basis[pr] = pc;
    }
    Tableau { t, basis, w, shift }
}

/// Full solution (both mixed strategies) via the simplex method.
pub fn solve(a: &[f64], rows: usize, cols: usize) -> GameSolution {
    let Tableau { t, basis, w, shift } = simplex(a, rows, cols);
    let obj = rows * w;
    let v = 1.0 / t[obj + w - 1];
    let mut col = vec![0.0; cols];
    for (i, &bv) in basis.iter().enumerate().take(rows) {
        if bv < cols {
            col[bv] = t[i * w + w - 1] * v;
        }
    }
    let row: Vec<f64> = (0..rows).map(|i| (t[obj + cols + i] * v).max(0.0)).collect();
    GameSolution { value: v - shift, row: normalize(row), col: normalize(col) }
}

fn normalize(mut p: Vec<f64>) -> Vec<f64> {
    let s: f64 = p.iter().sum();
    if s > 0.0 {
        p.iter_mut().for_each(|x| *x /= s);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_pennies() {
        let s = solve(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        assert!((s.value - 0.5).abs() < 1e-12);
        assert!((s.row[0] - 0.5).abs() < 1e-12 && (s.col[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rock_paper_scissors_shifted() {
        let a = [0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5];
        let s = solve(&a, 3, 3);
        assert!((s.value - 0.5).abs() < 1e-12);
        for p in s.row.iter().chain(&s.col) {
            assert!((p - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn saddle_point_matches_lp() {
        let a = [0.5, 0.6, 0.7, 0.4, 0.3, 0.9];
        assert!((value(&a, 2, 3) - solve(&a, 2, 3).value).abs() < 1e-12);
        assert!((value(&a, 2, 3) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn asymmetric_mixed() {
        // Row: [3,-1],[−2,1]  value = 1/7
        let s = solve(&[3.0, -1.0, -2.0, 1.0], 2, 2);
        assert!((s.value - 1.0 / 7.0).abs() < 1e-12);
        assert!((s.row[0] - 3.0 / 7.0).abs() < 1e-9);
        assert!((s.col[0] - 2.0 / 7.0).abs() < 1e-9);
    }
}
