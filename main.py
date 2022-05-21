import numpy as np


class SparseMatrix:

    def __init__(self, array: np.array) -> None:
        self.inter_represent: str = 'CSR'
        self.array = array
        self.number_of_rows, self.number_of_columns = np.shape(self.array)
        self.current_representation: dict
        self.to_csr()
        self.number_of_nonzero = len(self.current_representation)

    def to_csr(self):
        self.current_representation = dict()
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                if self.array[row][column] != 0:
                    self.current_representation[(row, column)] = self.array[row][column]
        self.inter_represent = "CSR"

    def to_csc(self):
        self.current_representation = dict()
        for column in range(self.number_of_columns):
            for row in range(self.number_of_rows):
                if self.array[row][column] != 0:
                    self.current_representation[(row, column)] = self.array[row][column]
        self.inter_represent = "CSC"

    def __str__(self) -> str:
        output = f"Representation: {self.inter_represent}\nNonzero = {self.number_of_nonzero}\n"
        for key, value in self.current_representation.items():
            output += f"{key} : {value}\n"
        return output


def main():
    x = np.array([[3, 0, 0, 6, 0, 3, 0],
                  [0, 4, 0, 0, 0, 0, 5],
                  [5, 6, 0, 0, 3, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [1, 0, 0, 9, 0, 0, 0]])
    print("Numpy array:")
    print(x)
    sparse_matrix = SparseMatrix(x)
    print(sparse_matrix)
    sparse_matrix.to_csc()
    print(sparse_matrix)


if __name__ == '__main__':
    main()
