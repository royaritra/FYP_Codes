import numpy as np
import matplotlib.pyplot as plt

hbar=1.06e-34
q=1.6e-19
m=0.1*9.1e-31
#Inputs
qh=q/hbar
a=2.5*1e-9
t0=(hbar**2)/(2*m*(a**2)*q);
NW=25
Np=1
L=np.zeros((Np,Np))
R=L;
L[0,0]=1
R[-1,-1]=1
zplus=1j*1e-12;  
Tcoh=np.array([])
E=np.array([])
Y=np.array([])
#Hamiltonian
al=4*t0
by=-t0
bx=-t0
alpha=np.kron(np.eye(NW),al)+np.kron(np.diag([1]*(NW-1),k=1),by)+np.kron(np.diag([1]*(NW-1),k=-1),np.transpose(np.conj(by)))
ap=np.arange(1,NW+1,1)
alpha=alpha+np.diag(ap*0)
EE=t0
ii=0
BS=np.arange(0,50,0.5)
for B in BS:
    ii=ii+1
    E=np.concatenate((E,[B]))
    ig0=(EE+zplus)*np.eye(NW)-alpha
    if ii==1:
        gs1=np.linalg.inv(ig0)
        gs2=np.linalg.inv(ig0)
    
    beta=np.kron(np.diag(np.exp(1j*qh*B*a*a*ap)),bx)
    H=np.kron(np.eye(Np),alpha)
    if Np>1:
        H=H+np.kron(np.diag([1]*(Np-1),k=1),beta)+np.kron(np.diag([1]*(Np-1),k=-1),np.transpose(np.conj(beta)))
    
    change=1
    while change>5e-5:
        Gs=np.linalg.inv(ig0-np.matmul(np.matmul(np.transpose(np.conj(beta)),gs1),beta))
        change=np.sum(np.sum(np.absolute(Gs-gs1),axis=0),axis=0) / np.sum(np.sum(np.absolute(gs1)+np.absolute(Gs),axis=0),axis=0)
        gs1=0.5*Gs+0.5*gs1
    
    sig1=np.matmul(np.matmul(np.transpose(np.conj(beta)),gs1),beta)
    sig1=np.kron(L,sig1)
    gam1=1j*(sig1-np.transpose(np.conj(sig1)))

    change=1
    while change>5e-5:
        Gs=np.linalg.inv(ig0-np.matmul(np.matmul(beta,gs2),np.transpose(np.conj(beta))))
        change=np.sum(np.sum(np.abs(Gs-gs2),axis=0),axis=0) / np.sum(np.sum(np.abs(gs2)+np.abs(Gs),axis=0),axis=0)
        gs2=0.5*Gs+0.5*gs2
    
    sig2=np.matmul(np.matmul(beta,gs2),np.transpose(np.conj(beta)))
    sig2=np.kron(R,sig2)
    gam2=1j*(sig2-np.transpose(np.conj(sig2)))
    
    G=np.linalg.inv((EE*np.eye(Np*NW))-H-sig1-sig2)
    Gn=np.matmul(np.matmul(G,gam1),np.transpose(np.conj(G)))
    A=1j*(G-np.transpose(np.conj(G)))
    V=np.real(np.diag(np.divide(Gn,A)))
    Tcoh=np.real(np.trace(np.matmul(gam1,np.matmul(G,np.matmul(gam2,np.transpose(np.conj(G)))))))
    TM=np.real(np.trace(np.matmul(gam2,Gn)))
    Y=np.concatenate((Y,[(V[0]-V[NW-1])/Tcoh]))



plt.plot(E,Y,'k', linewidth=3)
plt.xlabel('B-field(T) ---> ')
plt.ylabel('R_{xy} --->')

plt.show()

    