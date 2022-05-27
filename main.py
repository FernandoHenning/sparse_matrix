import numpy as np


class SparseMatrix:

    # TASK 1
    def __init__(self, array: np.array, tol=0.00000010) -> None:
        """
        This is the init method. When a new object is created this is the first thing that execute
        :param array: array to represent
        :param tol: tol value of the task 5
        """
        self.tol = tol
        self.inter_represent: str = 'CSR'  # TASK 1: initial type of representation
        self.array = array
        self.number_of_rows, self.number_of_columns = np.shape(self.array)  # Array size values ROW X COLUMN
        self.current_representation: dict  # Here I declare the current representation, but it doesn't have a value.
        self.number_of_nonzero = None  # TASK 2: the number of nonzero is None because this is just a declaration.
        self.to_csr()  # Call the CSR function and create the CSR representation.

    def to_csr(self):
        """
            This function creates the CSR representation
        """
        self.current_representation = dict()  # An empty dictionary were the CSR representation is going to be
        # represented.

        # We store the representation on a dictionary in the following form:
        # (Row index, Column index) = Nonzero value
        # This make easier access to the stored values.
        for row in range(self.number_of_rows):  # We start reading the values by row
            for column in range(self.number_of_columns):
                # TASK 5: the value must be greater that the tol value to be considered a nonzero value
                if self.array[row][column] > self.tol:
                    self.current_representation[(row, column)] = self.array[row][column]  # We store the value

        # Set the inter_representation to CSR
        self.inter_represent = "CSR"
        # The size of the dictionary corresponds to the number of nonzero
        self.number_of_nonzero = len(self.current_representation)

    # TASK 4
    def to_csc(self):
        """
        This function creates the CSC representation. We do the same thing as the CSR function.
        :return:
        """
        self.current_representation = dict()
        # We store the representation on a dictionary in the following form:
        # (Row index, Column index) = Nonzero value
        # This make easier access to the stored values.
        for column in range(self.number_of_columns):  # In this case we start reading values by column
            for row in range(self.number_of_rows):
                # TASK 5: the value must be greater that the tol value to be considered a nonzero value
                if self.array[row][column] > self.tol:
                    self.current_representation[(row, column)] = self.array[row][column]

        # Set the inter_representation to CSC
        self.inter_represent = "CSC"
        # The size of the dictionary corresponds to the number of nonzero
        self.number_of_nonzero = len(self.current_representation)

    def insert(self, i: int, j: int, a: float):
        """
        This function represents the task 3 which allows to change a particular element in the matrix.
        :param i: Row index
        :param j: Column index
        :param a: Value to by store
        :return: None
        """
        """
        Here we use a try-except structure to insert the value.
        In case the indexes are not in the presentation, the program will not stop. 
        This would just print an error saying that you are trying to access a value that does not 
        exist in the dictionary or an IndexError.
        """
        try:
            self.array[i][j] = a
        except IndexError as e:
            print(e)

        # We re-do the inter-representation with the new value.
        if self.inter_represent == "CSR":
            self.to_csr()
        else:
            self.to_csc()

    def __str__(self) -> str:
        """
        This is just a function to print the representation
        :return: A string representation of the inter-represent dictionary
        """
        output = f"Representation: {self.inter_represent}\nNonzero = {self.number_of_nonzero}\n"
        for key, value in self.current_representation.items():
            output += f"{key} : {value}\n"
        return output

    def __eq__(self, other):
        """
        Task 5: This function  check if two such matrices are (exactly) the same.
        :param other: The object to compare
        :return: True or false
        """
        if isinstance(other, (int, float)):  # Comparing the matrix with an int or float value
            raise "You are trying to compare a matrix with a integer of floating value."

        # If the size of both matrices is not the same we return false
        if self.number_of_rows != other.number_of_rows or self.number_of_columns != other.number_of_columns:
            # diff shape
            return False
        # If the number o nonzero is different we return false
        elif self.number_of_nonzero != other.number_of_nonzero:
            return False
        # If none of the above cases is true, we return True.
        return True

def main():
    """
    These examples were tests I performed. You can delete these and write your own.
    :return:
    """
    x = np.array([[3, 0, 0, 6, 0, 3, 0],
                  [0, 4, 0, 0, 0, 0, 5],
                  [5, 6, 0, 0, 3, 0.00000001, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [1, 0, 0, 9, 0, 0, 0]])

    x_sparse = SparseMatrix(x)
    print("Sparse matrix")
    print(x_sparse)
    x_sparse.to_csc()
    print(x_sparse)
    y = np.array([[3, 0, 0, 6, 0, 3, 0],
                  [0, 4, 0, 0, 0, 0, 5],
                  [5, 6, 0, 0, 3, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [1, 0, 0, 9, 0, 0, 0]])
    y_sparse = SparseMatrix(y)

    print("x == y ?", x_sparse == y_sparse)

    x_sparse.insert(0, 2, 1)
    print("New sparse matrix")
    print(x_sparse)
    print("x == y ?", x_sparse == y_sparse)


if __name__ == '__main__':
    main()
