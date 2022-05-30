
import numpy as np
import matplotlib.pyplot as plt

#constants

hbar=1.06*1e-34
q=1.6*1e-19
qh=q/hbar
m=0.1*9.1e-31
zplus =1j*1e-12                #small potential barrier(ie^-15)
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])
Tcoh=np.array([])
E=np.array([])
#inputs

a=2.5*1e-9
t0=(hbar*hbar)/(2*m*(a*a)*q)
eta=1e-11
NW=25
n=0
Np=2*n+1
NWp=NW*Np
L=np.zeros((Np,Np))
R=np.zeros((Np,Np))
L[0,0]=1
R[-1,-1]=1

#HAMILTONIAN

al=4*t0*np.eye(2)
by=-t0*np.eye(2)-(1j*eta/2/a)*sx  
bx=-t0*np.eye(2)+(1j*eta/2/a)*sy   
alpha=np.kron(np.eye(NW),al)+np.kron(np.diag([1]*(NW-1),k=1),by)+np.kron(np.diag([1]*(NW-1),k=-1),np.transpose(np.conj(by)))
beta=np.kron(np.diag([1]*(NW)),bx)
#print(beta)
H=np.kron(np.eye(Np),alpha)


if Np>1:
    H=H+np.kron(np.diag([1]*(Np-1),k=1),beta)+np.kron(np.diag([1]*(Np-1),k=-1),np.transpose(np.conj(beta)))

ii=0
A0=np.zeros((NWp,1))
N0=A0
Nx=A0
Ny=A0
Nz=A0

EES=np.arange(0.05,0.0504,0.005)*t0

for EE in EES:
    ii=ii+1
    ig0=(EE+zplus)*np.eye(2*NW)-alpha
    if ii==1:
        gs1=np.linalg.inv(ig0)
        gs2=np.linalg.inv(ig0)

    change=1
    while change>1e-4:
        Gs=np.linalg.inv(ig0-np.matmul(np.matmul(np.transpose(np.conj(beta)),gs1),beta))
        change=np.sum(np.sum(np.absolute(Gs-gs1),axis=0),axis=0) / np.sum(np.sum(np.absolute(gs1)+np.absolute(Gs),axis=0),axis=0)
        gs1=0.5*Gs+0.5*gs1
    
    sig1=np.matmul(np.matmul(np.transpose(np.conj(beta)),gs1),beta)
    sig1=np.kron(L,sig1)
    gam1=1j*(sig1-np.transpose(np.conj(sig1)))

    change=1
    while change>1e-4:
        Gs=np.linalg.inv(ig0-np.matmul(np.matmul(beta,gs2),np.transpose(np.conj(beta))))
        change=np.sum(np.sum(np.abs(Gs-gs2),axis=0),axis=0) / np.sum(np.sum(np.abs(gs2)+np.abs(Gs),axis=0),axis=0)
        gs2=0.5*Gs+0.5*gs2
    
    sig2=np.matmul(np.matmul(beta,gs2),np.transpose(np.conj(beta)))
    sig2=np.kron(R,sig2)
    gam2=1j*(sig2-np.transpose(np.conj(sig2)))

    G=np.linalg.inv((EE*np.eye(2*NWp))-H-sig1-sig2)
    A=1j*(G-np.transpose(np.conj(G)))
    Gn=np.matmul(np.matmul(G,gam1),np.transpose(np.conj(G)))
    Tcoh=np.concatenate((Tcoh,[np.real(np.trace(np.matmul(np.matmul(np.matmul(gam1,G),gam2),np.transpose(np.conj(G)))))]))

    S0=np.kron(np.eye(NWp),np.eye(2))
    Sx=np.kron(np.eye(NWp),sx)
    Sy=np.kron(np.eye(NWp),sy)
    Sz=np.kron(np.eye(NWp),sz)

    A0=A0+np.matmul(np.kron(np.eye(NWp),np.array([[1,1]])),np.diag(np.matmul(A,S0)))
    N0=N0+np.matmul(np.kron(np.eye(NWp),np.array([[1,1]])),np.diag(np.matmul(Gn,S0)))
    Nx=Nx+np.matmul(np.kron(np.eye(NWp),np.array([[1,1]])),np.diag(np.matmul(Gn,Sx)))
    Ny=Ny+np.matmul(np.kron(np.eye(NWp),np.array([[1,1]])),np.diag(np.matmul(Gn,Sy)))
    
    nz_a=np.matmul(np.kron(np.eye(NWp),np.array([[1,1]])),np.diag(np.matmul(Gn,Sz)+np.matmul(Sz,Gn)))
    Nz=Nz+nz_a.reshape(np.shape(nz_a)[0],1)
    
    E=np.concatenate((E,[EE/t0]))

F0=np.real(N0)
Fx=np.real(Nx)
Fy=np.real(Ny)
Fz=np.real(Nz)
Y=np.arange(0,NW-1,1)

#plt.plot(Y,F0[n*NW+1:(n+1)*NW],'k-o', linewidth=3)
#plt.plot(Y,Fx[n*NW+1:(n+1)*NW],'b', linewidth=3)
#plt.plot(Y,Fy[n*NW+1:(n+1)*NW],'k', linewidth=3)
plt.plot(Fz[n*NW+1:(n+1)*NW],Y,'k', linewidth=3)
plt.ylabel(' Width (nm) --> ')
plt.xlabel(' S_{z} (arbitrary units) --> ')

plt.show()





