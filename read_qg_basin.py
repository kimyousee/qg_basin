import numpy as np
import matplotlib.pyplot as plt
import os

### Script to read in data that was made already with qg_bas_io ###

FigDir = "figs"
if not os.path.exists(FigDir):
    os.mkdir(FigDir)

data = np.fromfile('storage/InputData')
eigVals = np.fromfile('storage/eigVals',dtype=np.complex128)
eigVecs = np.fromfile('storage/eigVecs',dtype=np.complex128)

H    = data[0]
beta = data[1]
f0   = data[2]
g    = data[3]
Lx   = data[4]
Ly   = data[5]
nEV  = int(data[6])
Nx   = int(data[7])
Ny   = int(data[8])
nmodes=int(data[9])

eigVecs = eigVecs.reshape([int(data[10]),nmodes])#,order='F')

for i in range(0,nmodes):
    mode = eigVecs[:,i]
    eigVal = eigVals[i]

    # If you have scalar-type complex for PETSc, use this code
    # If you have scalar-type real (default) for PETSc, 
    # get rid of all real. and imag., replace the ones that had .imag
    # with vi2d instead of vr2d

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

    fig = "figs/QG_Basin_m%d.eps" % i
    plt.savefig(fig, format='eps', dpi=1000)
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
