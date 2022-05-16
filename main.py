import numpy as np


class SparseMatrix:

    def __init__(self, array: np.array) -> None:
        self.array = array
        self.inter_represent: str = 'CSR'
        self.ROW_INDEX, self.COL_INDEX = np.nonzero(self.array)
        self.number_of_nonzero: int = len(self.COL_INDEX)  # Col index of nonzero values
        self.V = self.array[np.nonzero(self.array)]  # Non zero values
        print("ROW:", self.ROW_INDEX)
        print("COL:", self.COL_INDEX)
        print("VALUES:", self.V)


x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
test = SparseMatrix(x)
