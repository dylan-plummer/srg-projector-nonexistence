#!/usr/bin/env python3
"""Exact verification for the note "Two strongly regular graphs that do not exist".

Python 3 standard library only (fractions). Usage:  python3 verify.py

1. For srg(1275,546,281,198) and srg(1081,594,359,286): the spectrum, d = f, n = d(d+1)/2, the eigenvalues of
   H = G o G, H^{-1} = uI + vA + wJ, the pair-cell edge matrix E (affine in z = E_33), and the value of z forced by
   <R,R> = g^T H^{-1} g, for non-adjacent and for adjacent pairs.
2. Control: the triangular graphs T(m), 5 <= m <= 12 (n = d(d+1)/2 with d = m-1). For every pair, the cell
   equations hold for the actual edge counts and <R,R> = g^T H^{-1} g holds exactly.
"""
from fractions import Fraction as F
from itertools import combinations
from math import isqrt


def spectrum(n, k, lam, mu):
    D = (lam - mu) ** 2 + 4 * (k - mu)
    q = isqrt(D)
    assert q * q == D, 'non-integral eigenvalues'
    r, s = (lam - mu + q) // 2, (lam - mu - q) // 2
    f = ((n - 1) * (-s) - k) // (r - s)
    assert f * (r - s) == (n - 1) * (-s) - k
    return r, s, f


def data(n, k, lam, mu):
    """Embedding in the r-eigenspace: d, a, b, eigenvalues of H, and u, v, w with H^{-1} = uI + vA + wJ."""
    r, s, d = spectrum(n, k, lam, mu)
    a, b = F(-(1 + r), n - k - 1), F(r, k)
    c0, c1, c2 = 1 - a * a, b * b - a * a, a * a              # H = c0 I + c1 A + c2 J
    h0, hr, hs = c0 + c1 * k + c2 * n, c0 + c1 * r, c0 + c1 * s
    v = (1 / hr - 1 / hs) / (r - s)
    u = 1 / hs - v * s
    w = (1 / h0 - u - v * k) / n
    # check H (uI + vA + wJ) = I in the Bose-Mesner algebra, using A^2 = kI + lam A + mu (J - I - A)
    I_ = c0 * u + c1 * v * (k - mu)
    A_ = c0 * v + c1 * u + c1 * v * (lam - mu)
    J_ = c0 * w + c1 * v * mu + c1 * w * k + c2 * (u + v * k + w * n)
    assert (I_, A_, J_) == (1, 0, 0)
    return dict(r=r, s=s, d=d, a=a, b=b, h=(h0, hr, hs), uvw=(u, v, w))


def solve(Mrows):
    """Exact Gauss-Jordan on an augmented matrix; returns the unique solution or raises."""
    M = [row[:] for row in Mrows]
    nv = len(M[0]) - 1
    piv = []
    r = 0
    for c in range(nv):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        M[r] = [x / M[r][c] for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                M[i] = [x - M[i][c] * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    assert len(piv) == nv, 'not unique'
    assert all(all(x == 0 for x in row) for row in M[r:]), 'inconsistent'
    return [M[i][-1] for i in range(nv)]


CELLS = range(4)                                   # bit 0: adjacent to i, bit 1: adjacent to j
PAIRS = [(B, C) for B in CELLS for C in CELLS if B <= C]


def cell_sizes(n, k, lam, mu, adjacent):
    if adjacent:
        return [n - 2 - 2 * (k - 1) + lam, k - 1 - lam, k - 1 - lam, lam]
    return [n - 2 - 2 * k + mu, k - mu, k - mu, mu]


def cell_equations(n, k, lam, mu, adjacent):
    """Rows (coefficients on the 10 unknowns E_BC, B <= C; right-hand side) of the SRG cell equations."""
    sz = cell_sizes(n, k, lam, mu, adjacent)
    rows = []
    for B in CELLS:
        b0, b1 = B & 1, B >> 1
        row = [F(0)] * len(PAIRS)
        for C in CELLS:
            row[PAIRS.index((min(B, C), max(B, C)))] += 1
        rows.append(row + [F(sz[B] * (k - b0 - b1))])           # degree of x in C_B outside the anchors
        for t in (0, 1):
            bt, other = (B >> t) & 1, (B >> (1 - t)) & 1
            row = [F(0)] * len(PAIRS)
            for C in CELLS:
                if (C >> t) & 1:
                    row[PAIRS.index((min(B, C), max(B, C)))] += 1
            target = (lam if bt else mu) - (other if adjacent else 0)
            rows.append(row + [F(sz[B] * target)])               # common neighbours of x and anchor t
    return rows


def slack(D, n, k, lam, mu, adjacent, E):
    """<R,R> - g^T H^{-1} g for R = sym(p_i p_j^T), from the cell edge matrix E[(B,C)] (ordered counts)."""
    a, b, d = D['a'], D['b'], D['d']
    u, v, w = D['uvw']
    sz = cell_sizes(n, k, lam, mu, adjacent)
    gam = b if adjacent else a
    val = (a, b)
    gc = [val[B & 1] * val[B >> 1] for B in CELLS]
    Efull = lambda B, C: E[(min(B, C), max(B, C))]
    sum_g2 = 2 * gam ** 2 + sum(sz[B] * gc[B] ** 2 for B in CELLS)
    sum_Agg = sum(Efull(B, C) * gc[B] * gc[C] for B in CELLS for C in CELLS)
    sum_Agg += 2 * gam * sum(sz[B] * gc[B] * ((B & 1) + (B >> 1)) for B in CELLS)
    if adjacent:
        sum_Agg += 2 * gam * gam
    sum_g = F(n, d) * gam
    return (1 + gam ** 2) / 2 - (u * sum_g2 + v * sum_Agg + w * sum_g ** 2)


def forced_z(n, k, lam, mu, adjacent):
    D = data(n, k, lam, mu)
    rows = cell_equations(n, k, lam, mu, adjacent)
    i33 = PAIRS.index((3, 3))
    sols = []
    for z in (0, 1):                                            # E is affine in z = E_33
        fixed = [r[:i33] + r[i33 + 1:-1] + [r[-1] - r[i33] * z] for r in rows]
        x = solve(fixed)
        E = {p: x[j - (j > i33)] if j != i33 else F(z) for j, p in enumerate(PAIRS)}
        sols.append(slack(D, n, k, lam, mu, adjacent, E))
    s0, s1 = sols[0], sols[1] - sols[0]
    assert s1 != 0
    return -s0 / s1, D


def main():
    print('== The two parameter sets')
    for p in [(1275, 546, 281, 198), (1081, 594, 359, 286)]:
        n, k, lam, mu = p
        D = data(*p)
        assert n == D['d'] * (D['d'] + 1) // 2 and min(D['h']) > 0
        print(f'srg{p}: r = {D["r"]}, s = {D["s"]}, d = {D["d"]}, n = d(d+1)/2; a = {D["a"]}, b = {D["b"]}')
        print(f'   h = {tuple(str(x) for x in D["h"])}; (u, v, w) = {tuple(str(x) for x in D["uvw"])}')
        for adj in (False, True):
            z, _ = forced_z(*p, adj)
            print(f'   {"adjacent" if adj else "non-adjacent"} pair: E_33 = {z}  ->  ' +
                  ('NOT AN EVEN INTEGER: contradiction' if z.denominator != 1 or z % 2 else 'integral'))
            assert z.denominator != 1
        # Sims' relation for the 4-vertex parameters alpha = z_adj/2, beta = z_non/2 (BvM Prop. 8.16.1)
        al, be = forced_z(*p, True)[0] / 2, forced_z(*p, False)[0] / 2
        lhs, rhs = k * (F(lam * (lam - 1), 2) - al), be * (n - k - 1)
        assert lhs == rhs
        print(f'   Sims relation k(C(lam,2) - alpha) = beta(n-k-1): both sides {lhs}')
    print('== The same criterion on srg(351,140,73,44) (excluded earlier by Liao and Yu) and srg(703,182,81,35)')
    for p in [(351, 140, 73, 44), (703, 182, 81, 35)]:
        D = data(*p)
        assert p[0] == D['d'] * (D['d'] + 1) // 2 and min(D['h']) > 0 and D['h'][1] != D['h'][2]
        zs = [forced_z(*p, adj)[0] for adj in (False, True)]
        print(f'   srg{p}: forced edge counts beta = {zs[0] / 2}, alpha = {zs[1] / 2}  ->  ' +
              ('non-integral: excluded' if any(z.denominator != 1 for z in zs) else 'integral: no conclusion'))
    D8 = data(28, 12, 6, 4)
    assert D8['h'][1] == D8['h'][2] and D8['uvw'][1] == 0
    print('   T(8): h_r = h_s, v = 0: the identity does not involve z (no forced value)')
    print('== Control: triangular graphs T(m), all pairs')
    total = 0
    for m in range(5, 13):
        V = list(combinations(range(m), 2))
        n = len(V)
        A = [[int(len(set(x) & set(y)) == 1) for y in V] for x in V]
        k, lam, mu = 2 * (m - 2), m - 2, 4
        D = data(n, k, lam, mu)
        assert n == D['d'] * (D['d'] + 1) // 2 and min(D['h']) > 0
        for i, j in combinations(range(n), 2):
            adj = bool(A[i][j])
            cell = [(A[i][x]) | (A[j][x] << 1) for x in range(n)]
            E = {p: F(0) for p in PAIRS}
            for x in range(n):
                if x in (i, j):
                    continue
                for y in range(n):
                    if y in (i, j) or not A[x][y]:
                        continue
                    B, C = cell[x], cell[y]
                    if B <= C:
                        E[(B, C)] += 1
            for row in cell_equations(n, k, lam, mu, adj):         # actual counts satisfy the cell equations
                assert sum(c * E[p] for c, p in zip(row[:-1], PAIRS)) == row[-1]
            assert slack(D, n, k, lam, mu, adj, E) == 0              # and the projector identity
            total += 1
    print(f'   {total} pairs in T(5), ..., T(12): cell equations and <R,R> = g^T H^-1 g hold exactly')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
