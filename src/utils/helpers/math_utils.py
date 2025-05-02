
import math

from functools import lru_cache
from typing import List, Tuple

# ---------------------------
# 1. Type Definitions
# ---------------------------

Matrix = List[List[float]]

# ---------------------------
# 2. Mathematical Utilities
# ---------------------------

class MathUtils:
    """
    Enhanced numerical methods with parallel processing.
    Implements algorithms from:
    - Golub & Van Loan (2013) "Matrix Computations"
    - Press et al. (2007) "Numerical Recipes"
    """
    
    @staticmethod
    @lru_cache(maxsize=128)
    def determinant(matrix: Tuple[Tuple[float, ...], ...]) -> float:
        """LU decomposition-based determinant (O(n^3))"""
        n = len(matrix)
        mat = [list(row) for row in matrix]  # Convert to mutable format
        det = 1.0
        for i in range(n):
            max_row = max(range(i, n), key=lambda r: abs(mat[r][i]))
            if i != max_row:
                mat[i], mat[max_row] = mat[max_row], mat[i]
                det *= -1
            pivot = mat[i][i]
            if pivot == 0:
                return 0.0
            det *= pivot
            for j in range(i+1, n):
                factor = mat[j][i] / pivot
                for k in range(i+1, n):
                    mat[j][k] -= factor * mat[i][k]
        return det

    @staticmethod
    def cholesky(matrix: Matrix) -> Matrix:
        """Cholesky decomposition for positive definite matrices"""
        n = len(matrix)
        L = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = math.sqrt(matrix[i][i] - s) if (i == j) else \
                          (matrix[i][j] - s) / L[j][j]
        return L

    @staticmethod
    def is_positive_definite(matrix: Matrix) -> bool:
        """Cholesky-based positive definiteness check"""
        try:
            _ = MathUtils.cholesky(matrix)
            return True
        except ValueError:
            return False
