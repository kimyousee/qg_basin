import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

Print = PETSc.Sys.Print

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

def petscKron(A,B):
    dim = A.shape[0]*B.shape[0] # length of resulting matrix
    
    # Used to get indexes where values are non-zero
    Br,Bc = np.nonzero(B)
    Ar,Ac = np.nonzero(A)

    # Need to have values on first axis
    if Ar.shape[0] == 1:
        Ar = Ar.T
        Ac = Ac.T
    if Br.shape[0] == 1:
        Br = Br.T
        Bc = Bc.T

    # Distance between each 'block'
    n = B.shape[1]
    
    # create petsc resulting matrix
    K = PETSc.Mat().createAIJ([dim,dim])
    K.setFromOptions(); K.setUp()
    start,end = K.getOwnershipRange()

    for i in range(len(Ar)): # Go through each non-zero value in A
        # br,bc are used to track which 'block' we're in (in result matrix)
        br,bc = n*Ar[i], n*Ac[i]

        for j in range(len(Br)): # Go through non-zero values in B
            # kr,kc used to see where to put the number in K
            kr = (Br[j]+br).astype(np.int32)
            kc = (Bc[j]+bc).astype(np.int32)
            #Print(i,j,kr,kc)
            if start <= kr < end: # Make sure we're in the correct processor
                K[kr, kc] = A[Ar[i],Ac[i]] * B[Br[j],Bc[j]]

    K.assemble()
    return K

if __name__ == '__main__':
    opts = PETSc.Options()
    nEV = opts.getInt('nev', 10)
    Nx = opts.getInt('Nx',40)
    Ny = opts.getInt('Ny',40)

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

    Dxv  = petscKron(Dx,Iy)
    Lapv = petscKron(Ix, Dy2) + petscKron(Dx2,Iy)

    A = beta*Dxv
    B = f0**2/(g*H)*petscKron(Ix,Iy) - Lapv
    # start,end = B.getOwnershipRange()
    # for i in range(start,end):
    #   print B[i,:]
    E = SLEPc.EPS().create(comm=SLEPc.COMM_WORLD)
    E.setOperators(A,B)
    E.setDimensions(nEV,PETSc.DECIDE)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP); E.setFromOptions()
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)

    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    for i in range(0,1):
        eigVal = E.getEigenvalue(i)*1j
        #print eigVal.real + eigVal.imag*1j

        # If you have scalar-type complex for PETSc, use this code
        # If you have scalar-type real (default) for PETSc, 
        # get rid of all real. and imag., replace the ones that had .imag
        # with vi2d instead of vr2d

        E.getEigenvector(i,vr,vi) # Note: Both real and imaginary parts are in
                                  # vr if you have complex petsc.

        scatter, vrSeq = PETSc.Scatter.toZero(vr)
        im = PETSc.InsertMode.INSERT_VALUES
        sm = PETSc.ScatterMode.FORWARD
        scatter.scatter(vr,vrSeq,im,sm)

        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0:
            mode = np.empty(vr.getSize(),dtype='complex')

            for i in range(0,vrSeq.getSize()):
                mode[i] = vrSeq[i].real+vrSeq[i].imag*1j

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
            plt.show()

            # vr2d = np.reshape(vr,[Ny-1,Ny-1],order='F')
            # vi2d = np.reshape(vi,[Ny-1,Ny-1],order='F')

            # lvlr = np.linspace(vr2d.real.min(),vr2d.real.max(),20)
            # lvli = np.linspace(vr2d.imag.min(),vr2d.imag.max(),20)

            # plt.subplot(1,2,1)
            # plt.contourf(vr2d.real,levels=lvlr)
            # plt.title('real(psi)')

            # plt.subplot(1,2,2)
            # plt.contourf(vr2d[:].imag,levels=lvli)
            # plt.title('imag(psi)')

            # plt.show()

