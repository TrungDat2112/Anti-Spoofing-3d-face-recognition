import numpy as np


def mueller_stokes(S_in, theta=0):
     theta = np.deg2rad(theta)
     M = np.array(
          [[1, np.cos(2*theta), np.sin(2*theta), 0],
           [np.cos(2*theta), np.cos(2*theta)**2, np.cos(2*theta)*np.sin(2*theta), 0],
           [np.sin(2*theta), np.cos(2*theta)*np.sin(2*theta), np.sin(2*theta)**2, 0],
           [0, 0, 0, 0]]
     )
     # print(S_in.shape)
     S_out = np.einsum('ij,jklm->iklm',M, S_in)/2
     return S_out