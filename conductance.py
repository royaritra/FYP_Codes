#conductance/Aritra
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
t=1         #Hopping energy as 1eV
Np=20       #No of Particles in the balistic Conductor
L=np.zeros((20,20))
R=np.zeros((20,20))
L[0,0]=1
R[-1,-1]=1

Ldiag=np.diag([-t]*19,k=-1)  #"k" means first lower diagonal array elements as "-t"
Mdiag=np.diag([2*t]*20)      # all diagonal elements are "2*t"
Udiag=np.diag([-t]*19,k=1)   #"k" means first upper diagonal array elements as "-t"
H0=Ldiag+ Mdiag+ Udiag
#print(H0)
N1=4            #1st scatterer position
N2=7           # 2nd Scatterer position
UB1=2*t        #Scattering Potential for 1st Scatterer
UB2=0         #Scattering Potential for 2nd Scatterer, if you want to see results for only one sctterer, put UB2=0

##if you want to see results for only one sctterer, put UB2=0, if want to see for no scatterer , put Both UB1 and UB2 as zero

ieta =1j*1e-15                      #small potential barrier(ie^-15)

## Following range of EE Gives the best range to plot, I have taken 0.005 resolution for smoother graph, 
#You can change the interval as per requirements
#For the given problem, 1st interval was 0-4*t (EE=np.arange(0,4*t,0.005).T)
#2nd interval, E>4*t (EE=np.arange(4*t,7,0.005).T)
#3rd interval, E<0  (EE=np.arange(-4,0,0.005).T)

EE=np.arange(-0.5,4.5,0.005)     #number of elements arranged in row matrix;

print(EE)

#Adding Scattering potentials in Hamiltonian Matrix 

H0[N1,N1]=H0[N1,N1]+UB1
H0[N2,N2]=H0[N2,N2]+UB2
H=H0

#defining  lists to store all values
Trans=list()

for E in EE: 
    ck=(1-(E+ieta)/(2*t))       #1-E/(2*t)    
    ka=np.arccos(ck)    # cos inverse of "ck"
    ika=1j*ka           
    s1=-t*np.exp(ika)   
    s2=-t*np.exp(ika)   
    sig1=np.kron(L,s1)      #calculates sigma1
    sig1c=np.conj(sig1) 
    sig1ct=np.transpose(sig1c)  
    #print("sig1=",sig1)
    #print("sig1ct=",sig1ct)
    sig2=np.kron(R,s2)      #calculates sigma2
    sig2c=np.conj(sig2)
    sig2ct=np.transpose(sig2c)
    #print("sig2=",sig2)
    #print("sig2ct=",sig2ct)
    gamma1=1j*(sig1-sig1ct)     #Calculate Gamma 1
    gamma2=1j*(sig2-sig2ct)     #Calculate Gamma 2
    #print('Gamma 1: ', gamma1)
    #print('Gamma 2: ', gamma2)
    Green_GR=np.linalg.inv(E*np.eye(20)-H-sig1-sig2)       #Calculates Gr
    #print('Green_GR', Green_GR)
    Green_GA=np.conjugate(Green_GR)
    #print("Green_GA=",Green_GA)
    mul1=np.matmul(gamma1,Green_GR)     
    mul2=np.matmul(mul1,gamma2)        
    mul3=np.matmul(mul2,Green_GA)       
    #print('multiplication3: ', mul3)
    T=np.trace(mul3)        #Calculates Transmission Probablity
    Trans_E=T.real
    #Cond_G.append(Trans_E)
    Trans.append(Trans_E)
    #print("Trans_E=",Trans_E)
                           
#print(Trans)
plt.plot(Trans,EE) #Cond_g is x axis, EE is Y axis, alter to change the axis
plt.title('T(E) vs E for -0.5<E<4.5 with One scatterer')
plt.xlabel('T(E)')
plt.ylabel('E')
plt.show()
    


