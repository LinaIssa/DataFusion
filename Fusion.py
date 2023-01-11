import numpy as np
from abc import ABC, abstractmethod

class Cube(ABC): #si heritage mettre le nom de l heritage ceci est une classe abstraite
    '''
    Class that contains the cube

    '''

        def __init__(self, data:np.array, mask: np.array =None, **kwargs) -> None:
            if not isinstance(data, np.array):
                raise TypeError(f'data has type {type(data)} but it must be a numpy array ')
            self.data = data
            self.mask = mask
            self.dim  = data.shape

            # Flatten version of the data
            self.dataflat = self.flatten(data)
        @staticmethod
        def flatten( data: np.array, *args, **kwargs) -> np.array:
            r"""
            .. codeauthor:: Lina Issa - IRAP <lina.issa@irap.omp.eu>
            Transform a 3d array into a 2d flatten array

            : param data: input data
            :type data: numpy array
            :return: flatten array
            :rtype: np.array
            """



             x, y = np.arange(pix1), np.arange(pix2)
            MX, MY = np.meshgrid(y, x)
            X_cube = np.full((bands, pix1, pix2), np.nan)
            if bands in X.shape:

                if X.shape[0] == bands:
                    for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[1])):
                        X_cube[:, y, x] = X[:, z]
                if X.shape[1] == bands:
                    for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[0])):
                        X_cube[:, y, x] = X[z, :]
                return X_cube
            else:
                raise TypeError(
                    f'The given number of bands {bands} is in conflict with the shape of the given data {X.shape[0], X.shape[1]}.')


        @abstractmethod
        def regularisation(self, *args, **kwargs): # il faut le definir pour chaque classes qui heritent
            """
            performs get_linearsyst_split according to the type of regularization
            :param args:
            :param kwargs:
            :return:
            """
            return
        def postprocess(self, *args, **kwargs):
            return

        @abstractmethod


        def (self, *args, **kwargs):
            return
        def conjugate_gradient(self, *args, **kwargs):
            return
        def fusion(self, regP =[], regKw = {},
                         gradP=[], gradKw= {}): -> np.array
            regout= self.regularisation(           *regP , **regKw)
            out   = self.conjugategradient(regout, *gradP, **gradKw )


class CubeSob(Cube):
    def __init__(self,data:np.array, mask: np.array =None, **kwargs) -> None:
        super().__init__(data, mask, **kwargs) # appelle classe mere
        # truc supplementaires
    def regularisation(self, *args, **kwargs): # remplacer par les vrais parametres
        return
    def conjugate_gradient(self, *args, **kwargs):
        return

#self = objet lui meme
#cls renvoie a la classe,  pas de self en static
