import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs
import os

## qg_basin code using numpy only
## you can use either eig or eigs

def fd2(N):
    if N==0: D=0; x=1; return
    x = np.linspace(-1,1,N+1) #double check syntax
    h = 2./N
    e = np.ones(N+1)

    data = np.array([-1*e, 0*e, e])/(2*h)
    D = sp.spdiags(data, [-1, 0, 1], N+1,N+1)
    D = sp.csr_matrix(D)
    D[0, 0:2] = np.array([-1, 1])/h
    D[N, N-1:N+1] = np.array([-1, 1])/h

    D2 = sp.spdiags(np.array([e, -2*e, e])/h**2, [-1, 0, 1], N+1, N+1)
    D2 = sp.csr_matrix(D2)
    D2[0, 0:3] = np.array([1, -2, 1])/h**2
    D2[N, N-2:N+1] = np.array([1,-2,1])/h**2

    return D, D2, x

if __name__ == '__main__':
    OutpDir = "storage"
    if not os.path.exists(OutpDir):
        os.mkdir(OutpDir)

    data = open('storage/InputData','wb')
    eigVecsFile = open('storage/eigVecs','wb')
    eigValsFile = open('storage/eigVals','wb')

    Nx = 20
    Ny = 20
    nmodes = 5

    H    = 5e2               # Fluid Depth
    beta = 2e-11             # beta parameter
    f0   = 2*np.pi/(3600*24) # Mean Coriolis parameters
    g    = 9.81              # gravity
    Lx   = np.sqrt(2)*1e6    # Zonal Length
    Ly   = 1e6               # Meridional Length

    [Dx,Dx2,x]  = fd2(Nx);        [Dy,Dy2,y]  = fd2(Ny)
    x           = Lx/2*x;         y           = Ly/2*y
    Dx          = 2/Lx*Dx;        Dy          = 2/Ly*Dy
    Dx2         = (2/Lx)**2*Dx2;  Dy2         = (2/Ly)**2*Dy2

    xx,yy = np.meshgrid(x[1:Nx], y[1:Ny])
    xx = np.reshape(xx,(Nx-1)*(Ny-1), order='F')
    yy = np.reshape(yy,(Nx-1)*(Ny-1), order='F')
    Ix = np.eye(Nx-1)
    Iy = np.eye(Ny-1)

    Dx  = Dx[1:Nx,1:Nx]
    Dy2 = Dy2[1:Ny,1:Ny]
    Dx2 = Dx2[1:Nx,1:Nx]

    Dxv = sp.kron(Dx,Iy)
    Lapv = sp.kron(Ix,Dy2) + sp.kron(Dx2,Iy)

    A = beta*Dxv
    B = f0**2/(g*H)*sp.kron(Ix,Iy) - Lapv

    # Using eig
    eigVals, eigVecs = spalg.eig(A.todense(),B.todense())
    ind = (-np.imag(eigVals)).argsort() #get indices in descending order
    eigVecs = eigVecs[:,ind]
    eigVals = eigVals[ind]

    # Using eigs
    # eigVals, eigVecs = eigs(A,10,B,ncv=21,which='LI',maxiter=250)

    mode = np.empty(eigVecs.shape[0],dtype='complex')

    if nmodes > mode.shape: nmodes = mode.shape
    dataArr = np.array([H,beta,f0,g,Lx,Ly,eigVals.shape[0],Nx,Ny,nmodes,eigVecs.shape[0]])
    eigVals = eigVals[0:nmodes]
    eigVecs = eigVecs[:,0:nmodes]

    eigVecs.tofile(eigVecsFile)
    eigVals.tofile(eigValsFile)
    dataArr.tofile(data)
    eigVecsFile.close();eigValsFile.close;data.close()
