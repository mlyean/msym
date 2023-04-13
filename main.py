from collections import defaultdict
from sympy import factorint, gcd, gcdex, SparseMatrix, binomial, zeros, symbols, Poly, Matrix, Order
from sympy.ntheory.modular import crt
from bisect import bisect_left


def lift_ZmodnZ_star(n, d, a):
    # Given d divides n and a in (Z/dZ)^*, find a lift to (Z/nZ)^*
    u = 1
    for p in factorint(d).keys():
        while n % p == 0:
            u *= p
            n //= p
    return int(crt([u, n], [a, 1])[0])


class P1:

    def __init__(self, N):
        assert isinstance(N, int) and N >= 1
        self.N = N
        self._gcd = [gcdex(k, N) for k in range(N)]

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
        # Stein, Algorithm 8.29
        N = self.N
        u, v = p
        u %= N
        v %= N
        if u == 0:
            if self._gcd[v][2] == 1:
                return 0, 1
            raise ValueError
        s, _, g = self._gcd[u]
        if gcd(g, v) > 1:
            raise ValueError
        s = lift_ZmodnZ_star(N, N // g, s)
        u, v = g, (s * v) % N
        if g == 1:
            return 1, v
        v = min((v * t) % N for t in range(1, N, N // g) if gcd(t, N) == 1)
        return g, v

    def index(self, p):
        p0 = self.reduce(p)
        i = bisect_left(self._list, p0)
        if i != len(self._list) and self._list[i] == p0:
            return i
        raise ValueError


class ModularSymbols:

    def __init__(self, k, N):
        # M_k(Gamma0(N))
        assert k >= 2 and k % 2 == 0
        self.k = k
        self.N = N
        self._P1N = P1(N)
        self._msym = [(i, c, d) for i in range(k - 1) for (c, d) in self._P1N]

        ncols = len(self._msym)
        mat = SparseMatrix(2 * ncols, ncols, {})

        for row, (i, c, d) in enumerate(self._msym):
            mat[row, self.index((i, c, d))] += 1
            mat[row, self.index((k - 2 - i, d, -c))] += (-1)**i
            mat[row + ncols, self.index((i, c, d))] += 1

            for j in range(k - 2 - i + 1):
                mat[row + ncols, self.index(
                    (j, d,
                     -c - d))] += (-1)**(k - 2 + j) * binomial(k - 2 - i, j)
            for j in range(i + 1):
                mat[row + ncols,
                    self.index((k - 2 - i + j, -c - d,
                                c))] += (-1)**(k - 2 - i + j) * binomial(i, j)

        mat, piv = mat.rref()
        mat = mat[:len(piv), :]  # Remove zero rows
        free = self.free = tuple(k for k in range(ncols)
                                 if k not in piv)  # Indices of free generators

        # Construct relation matrix, could be simplified
        rel_mat = zeros(len(free), ncols)
        for e, col in enumerate(piv):
            for row in range(len(free)):
                rel_mat[row, col] = -mat[e, free[row]]
        for row, col in enumerate(free):
            rel_mat[row, col] = 1

        rel_mat_inv = SparseMatrix(ncols, len(free),
                                   {(row, col): 1
                                    for col, row in enumerate(free)})
        # rel_mat * rel_mat_inv == I
        self.rel_mat = rel_mat
        self.rel_mat_inv = rel_mat_inv

    def index(self, p):
        i, c, d = p
        return i * len(self._P1N) + self._P1N.index((c, d))

    def dim(self):
        return len(self.free)

    def cuspidal_subspace(self):
        k = self.k
        N = self.N
        bsym = BoundarySymbols(k, N)
        boundary_map = defaultdict(int)
        for col, e in enumerate(self.free):
            i, c, d = self._msym[e]
            a, b, g = gcdex(d, -c)
            assert g == 1
            boundary_map[(bsym.index((a, c)), col)] += 0**(k - 2 - i)
            boundary_map[(bsym.index((b, d)), col)] -= 0**i

        boundary_mat = SparseMatrix(len(bsym), len(self.free), boundary_map)

        return CuspidalModularSymbols(self, boundary_mat.nullspace())

    def right_action_mat(self, mat):
        k = self.k
        N = self.N
        ans = SparseMatrix(len(self._msym), len(self._msym), {})
        p, q, r, s = mat
        X, Y = symbols("X Y")
        for i in range(k - 1):
            pol = Poly((p * X + q * Y)**i * (r * X + s * Y)**(k - 2 - i), X, Y)
            for c, d in self._P1N:
                c1 = p * c + r * d
                d1 = q * c + s * d
                if gcd(c1, gcd(d1, N)) > 1:
                    continue
                for j in range(k - 1):
                    ans[self.index((j, c1, d1)),
                        self.index(
                            (i, c,
                             d))] = pol.coeff_monomial(X**j * Y**(k - 2 - j))
        return ans


class BoundarySymbols:
    # Stein, Algorithm 8.12

    def __init__(self, k, N):
        self.k = k
        self.N = N
        self._list = []

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def is_equiv(self, p, q):
        u1, v1 = p
        u2, v2 = q
        s1 = gcdex(u1, v1)[0]
        s2 = gcdex(u2, v2)[0]
        return (s1 * v2 - s2 * v1) % gcd(v1 * v2, self.N) == 0

    def index(self, p):
        for i, c in enumerate(self._list):
            if self.is_equiv(p, c):
                return i
        self._list.append(p)
        return len(self._list) - 1


def merel(n):
    # Merel, Proposition 20
    for a in range(1, n + 2):
        for d in range((n + a - 1) // a, n + 2 - a):
            for c in range(d):
                if c == 0:
                    if a * d == n:
                        for b in range(a):
                            yield [a, b, c, d]
                else:
                    if (a * d - n) % c == 0:
                        b = (a * d - n) // c
                        if b < a:
                            yield [a, b, c, d]


class CuspidalModularSymbols:

    def __init__(self, parent, basis):
        self._parent = parent
        self._basis = Matrix.hstack(*basis).reshape(len(parent.free),
                                                    len(basis))

    def dim(self):
        return self._basis.cols

    def T_matrix(self, n):
        parent = self._parent
        basis = self._basis
        l = merel(n)
        t = sum(map(lambda a: parent.right_action_mat(a), l),
                zeros(len(parent._msym), len(parent._msym)))
        return basis.LUsolve(parent.rel_mat * t * parent.rel_mat_inv * basis)


def cusp_forms(k, N, prec=10):
    # Computes a basis for S_k(Gamma0(N))
    m = ModularSymbols(k, N)
    s = m.cuspidal_subspace()
    d = s.dim()
    mat = zeros(d**2, prec - 1)
    for n in range(1, prec):
        mat[:, n - 1] = list(s.T_matrix(n))
    mat, piv = mat.rref()
    q = symbols('q')
    return [
        sum(mat[i, n - 1] * q**n for n in range(1, prec)) + Order(q**prec)
        for i in range(len(piv))
    ]
