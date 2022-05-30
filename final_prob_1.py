import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#t=1         #Hopping energy as 1eV
Np=50       #No of Particles in the balistic Conductor
N1=10       #Position of Injecting Probe
N2=20       #Position of Detecting Probe
ieta =1j*1e-12                #small potential barrier(ie^-15)
hbar=1.06*1e-34
q=1.6*1e-19
m=0.1*9.1*1e-31
a=2.5*1e-9
t0=(hbar*hbar)/(2*m*(a*a)*q)
print('check t0: ', t0)
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])
print(sx)
L=np.zeros((Np,Np))
R=np.zeros((Np,Np))
L[0,0]=1
R[-1,-1]=1
L1=np.zeros((Np,Np))
L1[N1-1,N1-1]=1
L1=L1*0.1
L2=np.zeros((Np,Np))
L2[N2-1,N2-1]=1
L1=L1*0.1
i=0
VV2=np.array([])
VVR=np.array([])
angle=np.array([])
I2=np.array([])
IL=np.array([])
IR=np.array([])
XX2=np.array([])


thetas=np.arange(0.5,4*np.pi,0.1*np.pi)
for theta in thetas:
    P1=0.7*np.array([0,0,1])
    P2=np.array([np.sin(theta),0,np.cos(theta)])

    #print(P1[2])
    Ldiag=np.diag([-t0]*(Np-1),k=-1)  #"k" means first lower diagonal array elements as "-t"
    Mdiag=np.diag([2*t0]*Np)      # all diagonal elements are "2*t"
    Udiag=np.diag([-t0]*(Np-1),k=1)   #"k" means first upper diagonal array elements as "-t"

    H=np.kron(Mdiag,np.eye(2))-np.kron(Ldiag,np.eye(2))-np.kron(Udiag,np.eye(2))

    EE=t0
    ck=(1-(EE+ieta)/(2*t0))       #1-E/(2*t)    
    ka=np.arccos(ck)    # cos inverse of "ck"
    
    ika=1j*ka           
    sL=-t0*np.exp(ika)*np.eye(2)   
    sR=-t0*np.exp(ika)*np.eye(2) 

    s1=-t0*np.exp(ika)*(np.eye(2)+P1[0]*sx+P1[1]*sy+P1[2]*sz)
    s2=-t0*np.exp(ika)*(np.eye(2)+P2[0]*sx+P2[1]*sy+P2[2]*sz)
    
    sigL=np.kron(L,sL)
    sigR=np.kron(R,sR)
    sig1=np.kron(L1,s1)
    sig2=np.kron(L2,s2)
    
    gamL=1j*(sigL-np.transpose(np.conj(sigL)))
    gamR=1j*(sigR-np.transpose(np.conj(sigR)))
    gam1=1j*(sig1-np.transpose(np.conj(sig1)))
    gam2=1j*(sig2-np.transpose(np.conj(sig2)))

    G=np.linalg.inv(((EE+ieta)*np.eye(2*Np))-H-sig1-sig2-sigL-sigR)
    G_a=np.transpose(np.conj(G))
    np.matmul(np.matmul(np.matmul(gam1,G),gamL),G_a)

    TM1L=np.real(np.trace(np.matmul(np.matmul(np.matmul(gam1,G),gamL),G_a)))
    TML1=np.real(np.trace(np.matmul(np.matmul(np.matmul(gamL,G),gam1),G_a)))
    Taa=np.array([[0,TM1L],[TML1,0]])
    #print(Taa)
    TM12=np.real(np.trace(np.matmul(np.matmul(np.matmul(gam1,G),gam2),G_a)))
    TM1R=np.real(np.trace(np.matmul(np.matmul(np.matmul(gam1,G),gamR),G_a)))
    TML2=np.real(np.trace(np.matmul(np.matmul(np.matmul(gamL,G),gam2),G_a)))
    TMLR=np.real(np.trace(np.matmul(np.matmul(np.matmul(gamL,G),gamR),G_a)))
    Tab=np.array([[TM12,TM1R],[TML2,TMLR]])
    TM21=np.real(np.trace(np.matmul(np.matmul(np.matmul(gam2,G),gam1),G_a)))
    TM2L=np.real(np.trace(np.matmul(np.matmul(np.matmul(gam2,G),gamL),G_a)))
    TMR1=np.real(np.trace(np.matmul(np.matmul(np.matmul(gamR,G),gam1),G_a)))
    TMRL=np.real(np.trace(np.matmul(np.matmul(np.matmul(gamR,G),gamL),G_a)))
    Tba=np.array([[TM21,TM2L],[TMR1,TMRL]])
    TM2R=np.real(np.trace(np.matmul(np.matmul(np.matmul(gam2,G),gamR),G_a)))
    TMR2=np.real(np.trace(np.matmul(np.matmul(np.matmul(gamR,G),gam2),G_a)))
    Tbb=np.array([[0,TM2R],[TMR2,0]])
    #print('check tbb')
    #print(Tbb)
    sum1=np.sum(Taa,axis=0)+np.sum(Tba,axis=0)
    Taa=np.diagflat(sum1)-Taa
    Tba=-Tba

    sum2=np.sum(Tab,axis=0)+np.sum(Tbb,axis=0)
    Tbb=np.diagflat(sum2)-Tbb
    Tab=-Tab
    
    V=np.matmul(np.matmul(-np.linalg.inv(Tbb),Tba),np.array([[1],[0]]))
    #print(np.shape(V))
    VV2=np.concatenate((VV2,[V[0,0]]))
    #print('vv2',VV2)
    VVR=np.concatenate((VVR,[V[1,0]]))
    angle=np.concatenate((angle,[theta/np.pi]))
    I2=np.concatenate((I2,[TM21]))
    IL=np.concatenate((IL,[TML1]))
    IR=np.concatenate((I2,[TMR1]))
    Gn=np.matmul(np.matmul(G,(gam1+V[0,0]*gam2+V[1,0]*gamR)),G_a)
    Gn=Gn[2*N2-2:2*N2,2*N2-2:2*N2]
    #Gn2=Gn[2*N2-2:2*N2+1]
    """ print(np.shape(Gn))    
    print(np.shape(Gn1))
    print(np.shape(Gn2))
    Gn=np.concatenate(Gn1,[Gn2]) """

    A=1j*(G-G_a)
    A=A[2*N2-2:2*N2,2*N2-2:2*N2]
    
    g2=1j*(s2-np.transpose(np.conj(s2)))
    
    XX2=np.concatenate((XX2,[np.real(np.trace(np.matmul(g2,Gn))/np.trace(np.matmul(g2,A)))]))

#print(VV2)
X2=VV2
XR=VVR
print(np.max(X2)-np.min(X2))
plt.plot(angle,X2)
plt.plot(angle,XX2,'ro')
plt.title('Spin Potential at Probe 2')
plt.xlabel('Angle')
plt.ylabel('Spin Potential')
plt.show()

#