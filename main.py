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
        self.row_index = []  # NEW: Now we store the row index values on a list.
        self.col_index = []  # NEW: Now we store the column index values on a list.
        self.values = []  # NEW: Now we store the nonzero values on a list.
        self.to_csr()
        self.number_of_nonzero = len(self.values)  # The number of nonzero values is equal to the size of the list

    def get_nonzero_values(self, n: int, m: int):
        """
        This function search for every nonzero value element on the matrix.

        :param n: This variable contains the size of the rows/columns of the matrix.
                  In case the CSR representation is needed, n corresponds to the number of rows;
                  if the representation is CSC, n corresponds to the number of columns.

        :param m: The value of this variable will always correspond to the opposite value of n.
                  If n is column, m will be row and vice versa.
        :return: A list of row index values, column index values and the nonzero values.
        """
        row_index = []
        col_index = []
        values = []
        if self.inter_represent == "CSR":
            for i in range(n):
                for j in range(m):
                    if self.array[i][j] > self.tol:
                        row_index.append(i)
                        col_index.append(j)
                        values.append(self.array[i][j])
        else:
            for i in range(n):
                for j in range(m):
                    if self.array[j][i] > self.tol:
                        row_index.append(i)
                        col_index.append(j)
                        values.append(self.array[j][i])
        return row_index, col_index, values

    def to_csr(self):
        """
        Here we get the nonzero values from the matrix. In this case, the N = number of rows and M = number of columns.
        :return:
        """
        n: int = self.number_of_rows
        m: int = self.number_of_columns
        self.inter_represent = "CSR"
        self.row_index, self.col_index, self.values = self.get_nonzero_values(n, m)

    # TASK 4
    def to_csc(self):
        """
        Here we get the nonzero values from the matrix. In this case, the N = number of columns and M = number of rows.
        :return:
        """
        n: int = self.number_of_columns
        m: int = self.number_of_rows
        self.inter_represent = "CSC"
        self.row_index, self.col_index, self.values = self.get_nonzero_values(n, m)

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

    def __eq__(self, other):
        """
        Task 5: This function  check if two such matrices are (exactly) the same.
        :param other: The object to compare
        :return: True or false
        """
        if isinstance(other, (int, float)):  # Comparing the matrix with an int or float value
            raise TypeError("You are trying to compare a matrix with a integer of floating value.")

        # If the size of both matrices is not the same we return false
        if self.number_of_rows != other.number_of_rows or self.number_of_columns != other.number_of_columns:
            # diff shape
            return False
        # If the number o nonzero is different we return false
        elif self.number_of_nonzero != other.number_of_nonzero:
            return False
        # If none of the above cases is true, we return True.
        for i in range(self.number_of_rows):
            for j in range(self.number_of_columns):
                if self.array[i][j] != other.array[i][j]:
                    return False
        return True

    def __str__(self) -> str:
        """
        This method is call when you print the object.
        :return: A string representation of the representation.
        """
        output = f"Representation : {self.inter_represent}\n"
        for (a, b, c) in zip(self.row_index, self.col_index, self.values):
            output += f"  ({a},{b})  =   {c}\n"
        return output

    # SECOND PART

    def __add__(self, other):
        from operator import add

        sum_list = []

        if not isinstance(other, SparseMatrix):
            # If not an SparseMatrix error throw an error
            raise TypeError("Elementwise addition is only available between objects of type SparseMatrix.")
        return SparseMatrix(np.array(list( map(add, self.array, other.array) )))

    def __mul__(self, other):
        from operator import mul
        sum_list = []
        if isinstance(other, (list, np.ndarray)):
            self.values = list(map(mul,self.values, other))
            return self.values
        else:
            raise TypeError("You only can multiply by a numpy.array or list objects of the same size..")


def get_testing_objects(size: int):
    from scipy import sparse
    size = 100
    x = y = np.zeros((size, size))

    x[0][0] = 2
    x[0][1] = 1
    x[1][0] = 1

    x[-1][-1] = 2
    x[-2][-1] = 1
    x[-1][-2] = 1

    x_sparse = SparseMatrix(x)

    y[0][0] = 2
    y[0][1] = 1
    y[1][0] = 1

    y[-1][-1] = 2
    y[-2][-1] = 1
    y[-1][-2] = 1
    y_sparse = SparseMatrix(y)

    x_scipy = sparse.csr_matrix(x)
    y_scipy = sparse.csr_matrix(y)

    vector = np.random.randint(0, 50, size)
    return x_sparse, y_sparse, x_scipy, y_scipy, vector

def test_class():
    import time
    x_sparse, y_sparse, _, _, vector = get_testing_objects(100)
    print("===========================================================")
    print("TESTING SparsMatrix CLASS WITH 100 ROWS")
    # Class performs
    # INSERTING
    start = time.perf_counter()
    x_sparse.insert(3, 3, 1)
    end = time.perf_counter()
    print(f"\tInserting a new element: {end - start} s")
    # SUMMING
    start = time.perf_counter()
    x_sparse + y_sparse
    end = time.perf_counter()
    print(f"\tSumming up two matrices: {end - start} s")
    # MULTIPLYING
    start = time.perf_counter()
    x_sparse * vector
    end = time.perf_counter()
    print(f"\tMultiplying a matrix with a vector: {end - start} s")

    x_sparse, y_sparse, _, _, vector = get_testing_objects(1000)
    print("TESTING SparsMatrix CLASS WITH 1000 ROWS")
    # Class performs
    # INSERTING
    start = time.perf_counter()
    x_sparse.insert(3, 3, 1)
    end = time.perf_counter()
    print(f"\tInserting a new element: {end - start} s")
    # SUMMING
    start = time.perf_counter()
    x_sparse + y_sparse
    end = time.perf_counter()
    print(f"\tSumming up two matrices: {end - start} s")
    # MULTIPLYING
    start = time.perf_counter()
    x_sparse * vector
    end = time.perf_counter()
    print(f"\tMultiplying a matrix with a vector: {end - start} s")

    x_sparse, y_sparse, _, _, vector = get_testing_objects(10000)
    print("TESTING SparsMatrix CLASS WITH 10000 ROWS")
    # Class performs
    # INSERTING
    start = time.perf_counter()
    x_sparse.insert(3, 3, 1)
    end = time.perf_counter()
    print(f"\tInserting a new element: {end - start} s")
    # SUMMING
    start = time.perf_counter()
    x_sparse + y_sparse
    end = time.perf_counter()
    print(f"\tSumming up two matrices: {end - start} s")
    # MULTIPLYING
    start = time.perf_counter()
    x_sparse * vector
    end = time.perf_counter()
    print(f"\tMultiplying a matrix with a vector: {end - start} s")
    print("END TESTING SparseMatrix CLASS")

def test_scipy_module():
    import time
    _, _, x_scipy, y_scipy, vector = get_testing_objects(100)
    print("===========================================================")
    print("TESTING Scipy MODULE WITH 100 ROWS")
    # Class performs
    # INSERTING
    start = time.perf_counter()
    y_scipy.tolil()[3] = 1
    end = time.perf_counter()
    print(f"\tInserting a new element: {end - start} s")
    # SUMMING
    start = time.perf_counter()
    x_scipy + y_scipy
    end = time.perf_counter()
    print(f"\tSumming up two matrices: {end - start} s")
    # MULTIPLYING
    start = time.perf_counter()
    x_scipy * vector
    end = time.perf_counter()
    print(f"\tMultiplying a matrix with a vector: {end - start} s")

    _, _, x_scipy, y_scipy , vector = get_testing_objects(1000)
    print("TESTING Scipy MODULE WITH 1000 ROWS")
    # Class performs
    # INSERTING
    start = time.perf_counter()
    y_scipy.tolil()[3] = 1
    end = time.perf_counter()
    print(f"\tInserting a new element: {end - start} s")
    # SUMMING
    start = time.perf_counter()
    x_scipy + y_scipy
    end = time.perf_counter()
    print(f"\tSumming up two matrices: {end - start} s")
    # MULTIPLYING
    start = time.perf_counter()
    x_scipy * vector
    end = time.perf_counter()
    print(f"\tMultiplying a matrix with a vector: {end - start} s")

    _, _, x_scipy, y_scipy , vector = get_testing_objects(10000)
    print("TESTING Scipy MODULE WITH 10000 ROWS")
    # Class performs
    # INSERTING
    start = time.perf_counter()
    y_scipy.tolil()[3] = 1
    end = time.perf_counter()
    print(f"\tInserting a new element: {end - start} s")
    # SUMMING
    start = time.perf_counter()
    x_scipy + y_scipy
    end = time.perf_counter()
    print(f"\tSumming up two matrices: {end - start} s")
    # MULTIPLYING
    start = time.perf_counter()
    x_scipy * vector
    end = time.perf_counter()
    print(f"\tMultiplying a matrix with a vector: {end - start} s")
    print("===========================================================")

def main():
    test_class()
    test_scipy_module()





if __name__ == '__main__':
    main()
