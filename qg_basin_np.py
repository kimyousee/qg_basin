import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs

## qg_basin code using numpy only
## you can use either eig or eigs

def fd2(N):
    if N==0: D=0; x=1; return
    x = np.linspace(-1,1,N+1) #double check syntax
    h = 2./N
    e = np.ones(N+1)

    data = np.array([-1*e, 0*e, e])/(2*h)
    D = sp.spdiags(data, [-1, 0, 1], N+1,N+1).todense()
    D[0, 0:2] = np.array([-1, 1])/h
    D[N, N-1:N+1] = np.array([-1, 1])/h
    sp.dia_matrix(D)

    D2 = sp.spdiags(np.array([e, -2*e, e])/h**2, [-1, 0, 1], N+1, N+1).todense()
    D2[0, 0:3] = np.array([1, -2, 1])/h**2
    D2[N, N-2:N+1] = np.array([1,-2,1])/h**2
    sp.dia_matrix(D2)
    return D, D2, x

if __name__ == '__main__':
    Nx = 40
    Ny = 40
    nmodes = 1
    
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

    Dxv = np.kron(Dx,Iy)
    Lapv = np.kron(Ix,Dy2) + np.kron(Dx2,Iy)

    A = beta*Dxv
    sp.dia_matrix(A)
    B = f0**2/(g*H)*np.kron(Ix,Iy) - Lapv
    sp.dia_matrix(B)

    # Using eig
    eigVals, eigVecs = spalg.eig(A,B)
    ind = (-np.imag(eigVals)).argsort() #get indices in descending order
    eigVecs = eigVecs[:,ind]
    eigVals = eigVals[ind]

    # # Using eigs
    # eigVals, eigVecs = eigs(A,10,B,ncv=21,which='LI',maxiter=250)

    mode = np.empty(eigVecs.shape[0],dtype='complex')

    for i in range(0,nmodes):
        mode = eigVecs[:,i]
        lvlr = np.linspace(mode.real.min(),mode.real.max(),20)
        lvli = np.linspace(mode.imag.min(),mode.imag.max(),20)
        mode = mode.reshape([Ny-1,Ny-1],order='F')

        plt.subplot(1,2,1)
        plt.contourf(mode.real,levels=lvlr)
        plt.title('real(psi)')

        plt.subplot(1,2,2)
        plt.contourf(mode.imag,levels=lvli)
        plt.title('imag(psi)')
        plt.colorbar(extend='both')
        
        plt.savefig('QG_Basin.eps', format='eps', dpi=1000)
        plt.show()

