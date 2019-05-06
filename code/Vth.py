import numpy as np
import matplotlib.pyplot as plt

#t_ox : A
#N_b : cm-3
#L, W, X_j : um
q=1.6e-19
k=8.62e-05
PHI_m=4.08 #eV, Gate material is Al
chi_s=4.05 
k_B=1.3806505e-23
epsilon_0=8.854e-12 #F/m
epsilon_ox=epsilon_SiO2=3.97
epsilon_Si=11.7 
T=300 #K

def n_i(T): #cm-3
	n_i=3.1e16*(T**1.5)*np.exp(-1.206/2/k/T)
	return n_i
def E_g(T): #eV
	E_g=1.16-7.02e-04*(T**2)/(1108+T)
	return E_g
def phi_f(N_b,T): #V
	phi_f=k_B*T/q*np.log(N_b/n_i(T))
	return phi_f
def PHI_s(N_b,T): #eV
	PHI_s=chi_s+E_g(T)/2+phi_f(N_b,T)
	return PHI_s
def C_ox(t_ox): #F/m2
	C_ox=epsilon_0*epsilon_ox/(t_ox*10**(-10))
	return C_ox
def PHI_ms(N_b,T):
	PHI_ms=-0.51-phi_f(N_b,T)
	return PHI_ms
def V_fb(N_b,T,Q_0,t_ox): #eV
	#V_fb=PHI_m-PHI_s(N_b,T)-Q_0/C_ox(t_ox)
	V_fb=PHI_ms(N_b,T)-Q_0/C_ox(t_ox)
	return V_fb 
def phi_s(N_b,T,V_sb): #V
	phi_s=2*phi_f(N_b,T)	
	return phi_s
def gamma(N_b,T,t_ox): #V1/2
	gamma=np.sqrt(2*epsilon_0*epsilon_Si*q*N_b*10**6)/C_ox(t_ox)
	return gamma

def V_T0(N_b,T,Q_0,t_ox):
	V_T0=V_fb(N_b,T,Q_0,t_ox)+2*phi_f(N_b,T)+gamma(N_b,T,t_ox)*np.sqrt(2*phi_f(N_b,T))
	return V_T0

def V_th_SPICE1(N_b,T,Q_0,t_ox,V_sb):
	V_th_SPICE1=V_T0(N_b,T,Q_0,t_ox)+gamma(N_b,T,t_ox)*(np.sqrt(2*phi_f(N_b,T)+V_sb)-np.sqrt(2*phi_f(N_b,T)))
	return V_th_SPICE1

def X_sd(N_b,phi_bi,V_sb):
	X_sd=np.sqrt(2*epsilon_0*epsilon_Si/q/N_b*10**6*(phi_bi+V_sb))#*10**6
	return X_sd
def X_dd(N_b,phi_bi,V_sb,V_ds):
	X_dd=np.sqrt(2*epsilon_0*epsilon_Si/q/N_b*10**6*(phi_bi+V_sb+V_ds))#*10**6
	return X_dd
def F_1(N_b,phi_bi,V_sb,V_ds,X_j,L):
	F_1=1-X_j/2/L*(np.sqrt(1+2*X_sd(N_b,phi_bi,V_sb)/X_j)+np.sqrt(1+2*X_dd(N_b,phi_bi,V_sb,V_ds)/X_j)-2)
	return F_1
def F_w(t_ox,G_w,W):
	F_w=np.pi/4*epsilon_0*epsilon_Si/C_ox(t_ox)*G_w/(W*10**(-6))
	return F_w
def V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,W,L,G_w):
	V_th_SPICE2=V_T0(N_b,T,Q_0,t_ox)-gamma(N_b,T,t_ox)*np.sqrt(2*phi_f(N_b,T))+gamma(N_b,T,t_ox)*F_1(N_b,phi_bi,V_sb,V_ds,X_j,L)*np.sqrt(2*phi_f(N_b,T)+V_sb)+F_w(t_ox,G_w,W)*(2*phi_f(N_b,T)+V_sb)
	return V_th_SPICE2


T=300
t_ox=np.arange(100,800,10)
plt.plot(t_ox,gamma(5*10**16,T,t_ox),label=r'$N_{b}=5\times10^{16} cm^{-3}$')
plt.plot(t_ox,gamma(1*10**16,T,t_ox),label=r'$N_{b}=1\times10^{16} cm^{-3}$')
plt.plot(t_ox,gamma(5*10**15,T,t_ox),label=r'$N_{b}=5\times10^{15} cm^{-3}$')
plt.plot(t_ox,gamma(1*10**15,T,t_ox),label=r'$N_{b}=1\times10^{15} cm^{-3}$')
plt.plot(t_ox,gamma(5*10**14,T,t_ox),label=r'$N_{b}=5\times10^{14} cm^{-3}$')
plt.legend()
plt.xlabel(r'$t_{ox}(\mathring{A})$')
plt.ylabel(r'$\gamma(V^{1/2})$')
plt.grid(True)
plt.show()


T=300
Q_0=0
V_sb=0
N_b=10**np.arange(13,18,0.1)
plt.plot(N_b,V_th_SPICE1(N_b,T,Q_0,650,V_sb),label=r'$t_{ox}=650\mathring{A}$')
plt.plot(N_b,V_th_SPICE1(N_b,T,Q_0,250,V_sb),label=r'$t_{ox}=250\mathring{A}$')
plt.plot(N_b,V_th_SPICE1(N_b,T,Q_0,150,V_sb),label=r'$t_{ox}=150\mathring{A}$')
plt.legend()
plt.xlabel(r'$N_{b}(cm^{-3})$')
plt.ylabel(r'$V_{th}(V)$')
plt.xscale('Log')
plt.grid(True)
plt.show()

T=300
Q_0=0
V_sb=0
phi_bi=0.8
V_ds=0
X_j=0.2
G_w=0.5
t_ox=200
N_b=2e16
X=np.arange(3,20,0.1)
plt.plot(X,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,X,20,G_w),label=r'$L=20\mu m$'+'    W changes')
plt.plot(X,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,20,X,G_w),label=r'$W=20\mu m$'+'    L changes')
plt.legend()
plt.xlabel('L  or  '+r'$W(\mu m)$')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()

N_b=2e16
T=300
Q_0=0
t_ox=150
phi_bi=0.8
V_ds=0
X_j=0.2
G_w=0.5
W=12.5
L=np.arange(0.4,6,0.01)
plt.plot(L,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,1,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=1V$')
plt.plot(L,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,0,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=0V$')
plt.legend()
plt.xlabel(r'$L(\mu m)$')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()


T=300
Q_0=0
t_ox=150
phi_bi=0.8
V_ds=0
V_sb=0
X_j=0.2
G_w=0.5
L=6
W=np.arange(0.4,6,0.01)
plt.plot(W,V_th_SPICE2(1.71e16,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,W,L,G_w),label=r'$N_{b}=1.71\times10^{16} cm^{-3}$')
plt.plot(W,V_th_SPICE2(1.56e16,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,W,L,G_w),label=r'$N_{b}=1.56\times10^{16} cm^{-3}$')
plt.plot(W,V_th_SPICE2(1.25e16,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,W,L,G_w),label=r'$N_{b}=1.25\times10^{16} cm^{-3}$')
plt.legend()
plt.xlabel(r'$W(\mu m)$')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()

N_b=2e16
T=300
Q_0=0
t_ox=150
phi_bi=0.8
X_j=0.2
G_w=0.5
W=12.5
L=2
V_ds=np.arange(0,5,0.01)
plt.plot(V_ds,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,3,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=3V$')
plt.plot(V_ds,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,0,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=0V$')
plt.legend()
plt.xlabel(r'$V_{ds}(V)$')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()

N_b=2e16
Q_0=0
t_ox=420
T=np.arange(0,160,0.1)
plt.plot(T,V_th_SPICE1(N_b,T,Q_0,t_ox,0),label=r'$V_{sb}=0V$')
plt.plot(T,V_th_SPICE1(N_b,T,Q_0,t_ox,3),label=r'$V_{sb}=3V$')
plt.legend()
plt.xlabel('T (C)')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()


'''
T=300
Q_0=0
V_sb=0
phi_bi=0.8
V_ds=0.1
X_j=0.2
W=25
L=6
G_w=0.5
N_b=10**np.arange(13,18,0.1)
plt.plot(N_b,V_th_SPICE2(N_b,T,Q_0,650,phi_bi,V_sb,V_ds,X_j,W,L,G_w),label=r'$t_{ox}=650\mathring{A}$')
plt.plot(N_b,V_th_SPICE2(N_b,T,Q_0,250,phi_bi,V_sb,V_ds,X_j,W,L,G_w),label=r'$t_{ox}=250\mathring{A}$')
plt.plot(N_b,V_th_SPICE2(N_b,T,Q_0,150,phi_bi,V_sb,V_ds,X_j,W,L,G_w),label=r'$t_{ox}=150\mathring{A}$')
plt.legend()
plt.xlabel(r'$N_{b}(cm^{-3})$')
plt.ylabel(r'$V_{th}(V)$')
plt.xscale('Log')
plt.grid(True)
plt.show()
'''

'''
N_b=10**np.arange(13,18,0.1)
t_ox=420
V_ds=0.1
T=300
V_sb=0
Q_0=0
phi_bi=0.8
X_j=0.2
G_w=0.5
plt.plot(np.sqrt(2*phi_f(N_b,T)+V_sb),V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,25,25,G_w),label=r'$W/L=25/25(\mu m)$')
plt.plot(np.sqrt(2*phi_f(N_b,T)+V_sb),V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,25,6,G_w),label=r'$W/L=25/6(\mu m)$')
plt.plot(np.sqrt(2*phi_f(N_b,T)+V_sb),V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,25,4,G_w),label=r'$W/L=25/4(\mu m)$')
plt.legend()
plt.xlabel(r'$\sqrt{2\phi_f+V_{sb}}(V^{1/2})$')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()
'''
'''
N_b=2e16
T=300
Q_0=0
t_ox=420
phi_bi=0.8
V_ds=0
X_j=0.2
G_w=0.5
W=25
L=np.arange(0.2,6,0.01)
plt.plot(L,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,2,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=2V$')
plt.plot(L,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,1.5,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=1.5V$')
plt.plot(L,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,1,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=1V$')
plt.plot(L,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,0.5,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=0.5V$')
plt.plot(L,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,0,V_ds,X_j,W,L,G_w),label=r'$V_{sb}=0V$')
plt.legend()
plt.xlabel(r'$L(\mu m)$')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()
'''

'''
N_b=2e16
T=300
Q_0=0
t_ox=150
phi_bi=0.8
V_ds=0.1
X_j=0.2
G_w=0.5
W=25
L=6
V_sb=np.arange(0.4,18,0.01)
plt.plot(V_sb,V_th_SPICE2(N_b,T,Q_0,t_ox,phi_bi,V_sb,V_ds,X_j,W,L,G_w))
plt.xlabel(r'$V_{sb}(V)$')
plt.ylabel(r'$V_{th}(V)$')
plt.grid(True)
plt.show()
'''
