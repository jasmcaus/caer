# Pulled from Scikit-Learn's official Github Repo (18 Sep 2020) to speed up 'caer' package import speeds (since this was the only method referenced from sklearn)

MAXPRINT = 50

class spmatrix(object):
    """ This class provides a base class for all sparse matrices.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    """

    def __init__(self, maxprint=MAXPRINT):
        self._shape = None
        if self.__class__.__name__ == 'spmatrix':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        self.maxprint = maxprint

    # Any sparse matrix format deriving from spmatrix must define one of
    # tocsr or tocoo. The other conversion methods may be implemented for
    # efficiency, but are not required.
    def tocsr(self, copy=False):
        """Convert this matrix to Compressed Sparse Row format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant csr_matrix.
        """
        return self.tocoo(copy=copy).tocsr(copy=False)


    def tocoo(self, copy=False):
        """Convert this matrix to COOrdinate format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant coo_matrix.
        """
        return self.tocsr(copy=False).tocoo(copy=copy)