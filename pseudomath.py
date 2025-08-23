# ========== CONSTANTS ==========

e = 2.718281828459045

# ========== RANDOMS ==========

RANDOM_SEED = 42

def lcg(seed):
    a = 1664525
    c = 1013904223
    m = 2**32
    while True:
        seed = (a * seed + c) % m
        yield seed / m

# ========== MATRICES ==========

class Matrix:
    # X = Matrix()
    def __init__(self, matrix: list[list[float]]):
        self.data = matrix

    # X.dim()
    def dim(self):
        return len(self.data), len(self.data[0])

    # X.flatten()
    def flatten(self):
        flattened = [[elem] for row in self.data for elem in row]
        return Matrix(flattened)

    # X.transpose()
    def transpose(self):
        m, n = self.dim()
        transposed = [[self.data[i][j] for i in range(m)] for j in range(n)]
        return Matrix(transposed)

    # X[i, j]
    def __getitem__(self, item: tuple[int, int]):
        return self.data[item[0]][item[1]]

    # X[i, j] = c
    def __setitem__(self, key, value):
        if type(key) is not tuple and len(key) != 2:
            raise IndexError('__getitem__ requires 2 indices')
        self.data[key[0]][key[1]] = value

    # X + Y
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[i + other for i in row] for row in self.data])
        elif isinstance(other, Matrix):
            if other.dim() == self.dim():
                m, n = self.dim()
                return Matrix([[self[i, j] + other[i, j] for j in range(n)] for i in range(m)])
            raise IndexError('Matrices need to have the same dimensions')
        raise ValueError('Matrix addition only supports types int, float, or Matrix')

    # X += Y
    def __iadd__(self, other):
        m, n = self.dim()
        if isinstance(other, (int, float)):
            for i in range(m):
                for j in range(n):
                    self[i, j] += other
        elif isinstance(other, Matrix):
            if other.dim() != self.dim():
                raise IndexError("Matrices need to have the same dimensions")
            for i in range(m):
                for j in range(n):
                    self[i, j] += other[i, j]
        return self

    # X + Y
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[i - other for i in row] for row in self.data])
        elif isinstance(other, Matrix):
            if other.dim() == self.dim():
                m, n = self.dim()
                return Matrix([[self[i, j] - other[i, j] for j in range(n)] for i in range(m)])
            raise IndexError('Matrices need to have the same dimensions')
        raise ValueError('Matrix subtraction only supports types int, float, or Matrix')

    # X += Y
    def __isub__(self, other):
        m, n = self.dim()
        if isinstance(other, (int, float)):
            for i in range(m):
                for j in range(n):
                    self[i, j] -= other
        elif isinstance(other, Matrix):
            if other.dim() != self.dim():
                raise IndexError("Matrices need to have the same dimensions")
            for i in range(m):
                for j in range(n):
                    self[i, j] -= other[i, j]
        return self

    # X * c or X * Y (elementwise)
    def __mul__(self, other):
        m, n = self.dim()
        if isinstance(other, (int, float)):
            return Matrix([[i * other for i in row] for row in self.data])
        elif isinstance(other, Matrix):
            if other.dim() != self.dim():
                raise IndexError("Matrices must have the same dimensions for elementwise multiplication")
            return Matrix([[self[i, j] * other[i, j] for j in range(n)] for i in range(m)])
        raise ValueError('Matrix multiplication only supports int, float, or Matrix (elementwise)')

    # X *= c or X *= Y (elementwise)
    def __imul__(self, other):
        m, n = self.dim()
        if isinstance(other, (int, float)):
            for i in range(m):
                for j in range(n):
                    self[i, j] *= other
        elif isinstance(other, Matrix):
            if other.dim() != self.dim():
                raise IndexError("Matrices must have the same dimensions for elementwise multiplication")
            for i in range(m):
                for j in range(n):
                    self[i, j] *= other[i, j]
        return self

    # X / c
    def __div__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[i / other for i in row] for row in self.data])
        raise ValueError('Matrix addition only supports types int, float, or Matrix')

    # X /= c
    def __itruediv__(self, other):
        m, n = self.dim()
        if isinstance(other, (int, float)):
            for i in range(m):
                for j in range(n):
                    self[i, j] /= other
        return self

    # X @ Y
    def __matmul__(self, other):
        m, n = self.dim()
        n_other, p = other.dim()
        if n != n_other:
            raise IndexError("Matrix size mismatch")

        other_cols = [[other.data[k][j] for k in range(n)] for j in range(p)]

        result_data = [
            [sum(map(lambda x_y: x_y[0] * x_y[1], zip(self.data[i], other_cols[j])))
             for j in range(p)]
            for i in range(m)
        ]

        return Matrix(result_data)

    # str(X)
    def __str__(self):
        m, n = self.dim()
        return f'Matrix ({m}x{n}):\n' + '\n'.join([str(line) for line in self.data])

    # X
    def __repr__(self):
        m, n = self.dim()
        return f'Matrix ({m}x{n})'

    @classmethod
    def const(cls, c: int | float, dims: tuple[int, int]):
        m, n = dims
        lst = [[float(c) for _ in range(n)] for _ in range(m)]
        return cls(lst)

    @classmethod
    def rand(cls, dims: tuple[int, int], bounds=(-2, 2)):
        m, n = dims
        low, high = bounds
        mat = cls.const(0, (m, n))
        gen = lcg(RANDOM_SEED)
        for i in range(m):
            for j in range(n):
                r = next(gen)
                mat[i, j] = low + r * (high - low)
        return mat

    @classmethod
    def like(cls, c: int | float, ref_matrix):
        return cls.const(c, ref_matrix.dim())

# ========== ACTIVATION FUNCTIONS ==========

def sigmoid(matrix: Matrix):
    m, n = matrix.dim()
    for i in range(m):
        for j in range(n):
            matrix[i, j] = 1 / (1 + e ** (-matrix[i, j]))
    return matrix

def relu(matrix: Matrix):
    m, n = matrix.dim()
    for i in range(m):
        for j in range(n):
            if matrix[i, j] < 0:
                matrix[i, j] = 0
    return matrix

def relu_prime(matrix: Matrix):
    m, n = matrix.dim()
    for i in range(m):
        for j in range(n):
            if matrix[i, j] > 0:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    return matrix

def softmax(matrix: Matrix):
    m, n = matrix.dim()
    vector = [matrix[i, 0] for i in range(m)]
    max_n = max(vector)
    for i, n in enumerate(vector):
        vector[i] = e**(n - max_n)
    summed = sum(vector)
    for i, n in enumerate(vector):
        vector[i] = n/summed
    return Matrix([[i] for i in vector])
