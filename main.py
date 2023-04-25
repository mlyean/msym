from bisect import bisect_left
from collections import defaultdict
from itertools import chain
from math import comb
from sympy import SparseMatrix, Matrix, Order, Poly, zeros, symbols


def gcdex(a, b):
    """Return a tuple (x, y, g) where g = gcd(a, b) and a*x + b*y == g."""
    if b == 0:
        if a < 0:
            return -1, 0, -a
        return 1, 0, a
    q, r = divmod(a, b)
    x, y, g = gcdex(b, r)
    return y, x - y * q, g


def gcd(a, b):
    return gcdex(a, b)[2]


def lift_ZmodnZ_star(n, d, a):
    """Given a divisor d of n and a unit a modulo d, lift a to a unit modulo n."""
    u, v = 1, n
    g = gcd(v, d)
    while g > 1:
        u *= g
        v //= g
        g = gcd(v, g)
    x, y, _ = gcdex(u, v)
    return (u * x + a * y * v) % n


class P1:
    """Projective line over Z/NZ, P^1(Z/NZ)

    Compute and store a list of (inequivalent) representatives for P^1(Z/NZ).

    See Stein, Algorithm 8.32.
    """

    def __init__(self, N):
        assert isinstance(N, int) and N >= 1
        self.N = N

        # Compute representatives
        tmp = set()
        for u in range(N):
            for v in range(N):
                try:
                    tmp.add(self.reduce((u, v)))
                except ValueError:
                    continue

        self._list = sorted(tmp)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def reduce(self, p):
        """Compute the canonical form of a pair p.

        This is an implementation of Stein, Algorithm 8.29.
        """
        N = self.N
        u, v = p
        u %= N
        v %= N
        if u == 0:
            if gcd(N, v) == 1:
                return 0, 1
            raise ValueError
        _, s, g = gcdex(N, u)
        if gcd(g, v) > 1:
            raise ValueError
        s = lift_ZmodnZ_star(N, N // g, s)
        u, v = g, (s * v) % N
        if g == 1:
            return 1, v
        v = min((v * t) % N for t in range(1, N, N // g) if gcd(N, t) == 1)
        return g, v

    def index(self, p):
        """Return the index of the pair p in the list."""
        p0 = self.reduce(p)
        i = bisect_left(self._list, p0)  # Find the index by binary search
        if i != len(self._list) and self._list[i] == p0:
            return i
        raise ValueError


class ModularSymbols:
    """Modular symbols of even weight k >= 2 for Gamma0(N), M_k(Gamma0(N)).

    Compute a list of Manin symbols, their relations and free generators.
    """

    def __init__(self, k, N):
        assert k >= 2 and k % 2 == 0
        self.k = k
        self.N = N
        self.P1N = P1(N)
        self.msym = [(i, c, d) for i in range(k - 1) for c, d in self.P1N]

        ncols = len(self.msym)
        mat = SparseMatrix(2 * ncols, ncols, {})

        for row, (i, c, d) in enumerate(self.msym):
            mat[row, self.index((i, c, d))] += 1
            mat[row, self.index((k - 2 - i, d, -c))] += (-1)**i
            mat[row + ncols, self.index((i, c, d))] += 1

            for j in range(k - 2 - i + 1):
                col = self.index((j, d, -c - d))
                mat[row + ncols, col] += (-1)**(k - 2 + j) * comb(k - 2 - i, j)
            for j in range(i + 1):
                col = self.index((k - 2 - i + j, -c - d, c))
                mat[row + ncols, col] += (-1)**(k - 2 - i + j) * comb(i, j)

        mat, piv = mat.rref()
        # Indices of free generators
        self.free = tuple(k for k in range(ncols) if k not in piv)

        # Construct relation matrix
        self.rel_mat = zeros(len(self.free), ncols)
        for e, col in enumerate(piv):
            for row, j in enumerate(self.free):
                self.rel_mat[row, col] = -mat[e, j]
        for row, col in enumerate(self.free):
            self.rel_mat[row, col] = 1

    def index(self, p):
        """Return the index of the Manin symbol p in the list."""
        i, c, d = p
        return i * len(self.P1N) + self.P1N.index((c, d))

    def dim(self):
        """Return the dimension."""
        return len(self.free)

    def cuspidal_subspace(self):
        """Return the subspace of cuspidal modular symbols."""
        k = self.k
        N = self.N
        bsym = BoundarySymbols(self, k, N)
        boundary_map = defaultdict(int)
        for col, e in enumerate(self.free):
            i, c, d = self.msym[e]
            a, b, g = gcdex(d, -c)
            assert g == 1
            boundary_map[(bsym.index((a, c)), col)] += 0**(k - 2 - i)
            boundary_map[(bsym.index((b, d)), col)] -= 0**i

        boundary_mat = SparseMatrix(len(bsym), self.dim(), boundary_map)

        return CuspidalModularSymbols(self, boundary_mat.nullspace())

    def right_action_mat(self, mat):
        """Return the matrix corresponding to the right action of mat."""
        k = self.k
        N = self.N
        ans = SparseMatrix(len(self.msym), self.dim(), {})
        p, q, r, s = mat

        # p1[i][j] is the coefficient of X^j*Y^(i-j) in (p*X+q*Y)^i
        # p2[i][j] is the coefficient of X^j*Y^(i-j) in (r*X+s*Y)^i
        p1 = [[0] * i for i in range(1, k)]
        p2 = [[0] * i for i in range(1, k)]
        p1[0][0] = 1
        p2[0][0] = 1
        for i in range(k - 2):
            for j in range(i + 1):
                p1[i + 1][j] += q * p1[i][j]
                p1[i + 1][j + 1] += p * p1[i][j]
                p2[i + 1][j] += s * p2[i][j]
                p2[i + 1][j + 1] += r * p2[i][j]

        for col, idx in enumerate(self.free):
            i, c, d = self.msym[idx]
            c1 = (p * c + r * d) % N
            d1 = (q * c + s * d) % N
            if gcd(N, gcd(c1, d1)) > 1:
                continue
            for j in range(k - 1):
                row = self.index((j, c1, d1))
                ans[row, col] = sum(p1[i][u] * p2[k - 2 - i][j - u]
                                    for u in range(max(0, i + j - (k - 2)),
                                                   min(i, j) + 1))
        return self.rel_mat * ans


class BoundarySymbols:
    """Boundary symbols of even weight k >= 2 for Gamma0(N), B_k(Gamma0(N)).

    See Stein, Algorithm 8.12.
    """

    def __init__(self, parent, k, N):
        self._parent = parent
        self.k = k
        self.N = N
        self._list = []

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def is_equiv(self, p, q):
        """Check if two boundary symbols are equivalent."""
        u1, v1 = p
        u2, v2 = q
        s1 = gcdex(u1, v1)[0]
        s2 = gcdex(u2, v2)[0]
        return (s1 * v2 - s2 * v1) % gcd(self.N, (v1 * v2) % self.N) == 0

    def index(self, p):
        """Return the index of p in the list."""
        for i, c in enumerate(self._list):
            if self.is_equiv(p, c):
                return i
        self._list.append(p)
        return len(self._list) - 1


def merel(n):
    """Compute the matrices in Merel's set X."""
    for a in range(1, n + 1):
        for d in range((n + a - 1) // a, n + 2 - a):
            bc = a * d - n
            if bc == 0:
                for b in range(a):
                    yield a, b, 0, d
                for c in range(1, d):
                    yield a, 0, c, d
            else:
                for b in range((bc - 1) // (d - 1) + 1, a):
                    if bc % b == 0:
                        yield a, b, bc // b, d


class CuspidalModularSymbols:
    """Cuspidal modular symbols of weight k for Gamma0(N), S_k(Gamma0(N))."""

    def __init__(self, parent, basis):
        self._parent = parent
        self._basis = Matrix.hstack(*basis).reshape(parent.dim(), len(basis))

    def dim(self):
        """Return the dimension."""
        return self._basis.cols

    def T_matrix(self, n):
        """Return the matrix corresponding to the Hecke operator T_n."""
        parent = self._parent
        basis = self._basis
        t = sum((parent.right_action_mat(a) for a in merel(n)),
                zeros(parent.dim(), parent.dim()))
        return basis.LUsolve(t * basis)


def cusp_forms(k, N, prec=10):
    """Compute a basis for the cusp forms of even weight k >= 2, level Gamma0(N)
    up to the specified precision prec."""
    m = ModularSymbols(k, N)
    s = m.cuspidal_subspace()
    d = s.dim()
    mat = zeros(d**2, prec - 1)
    for n in range(1, prec):
        mat[:, n - 1] = s.T_matrix(n).reshape(d**2, 1)
    mat, piv = mat.rref()
    q = symbols('q')
    basis = [
        Poly(chain(reversed(mat[i, :]), [0]), q).as_expr() + Order(q**prec)
        for i in range(len(piv))
    ]
    for _ in range(d // 2 - len(basis)):
        basis.append(Order(q**prec))
    return basis
