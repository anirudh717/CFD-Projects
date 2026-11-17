#============================================================
#CS-501 - Project
#Solver Code (CHT) - Version 1
#============================================================

#ASCII  Art of the Geometry and flow domain:


'''                                     

                                       insulated edge
                    |========================================================|
    insulated edge  |                       Solid                            |  insulated edge
                    |========================================================|   
                    |----->                                                  |   
                    |-------->                      FLUID Domain             |               
           inlet    |---------->                                             |  outlet     
                    |-------->                                               |
                    |----->                                                  |   
                    |========================================================|  
    insulated edge  |                       Solid                            |  insulated edge
                    |========================================================|
                    ‾‾‾‾'-._  ^ ^ ^ ^ ^ ^ ^ ^ ^ -'‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                            '.| | | | | | | | |'
                              \ | | | | | | |/
                               \  | | | | | /
                                \ | | | | |/
                                 '. | | |.'
                                   '._.'
                                Gaussian Heat Flux
                                     |
                                     x_c
                    ------------------------------------> x
                                           
'''

#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pdb import set_trace
import sys

#======================
# GLOBAL PLOT Settings:
#======================
plt.rcParams.update({
    'font.size':          12,       
    'font.weight':        'bold',   
    'axes.labelweight':   'bold', 
    'axes.titleweight':   'bold',  
    'axes.labelsize':     13,
    'axes.titlesize':     14,
    'axes.linewidth':     2,        
    'xtick.major.width':  2,        
    'ytick.major.width':  2,
    'xtick.major.size':   6,        
    'ytick.major.size':   6,
    'xtick.direction':    'in',     
    'ytick.direction':    'in',
    'lines.linewidth':    2,       
    'legend.frameon':     True,
    'legend.edgecolor':   'black',
    'savefig.dpi':        200,      
    'savefig.bbox':       'tight',  
})

#=================
#FLAGS
#=================
SOLVE = True
PLOT_GRID=False
PLOT=True


#=====================
#GRID
#=====================
#Grid specifications
Nx = 100
Ny = 50

#grid bounds
x_min, x_max = -1.0, 1.0      
y_min, y_max = -0.5, 0.5 

#interface
y_int = 0.3     

#grid step sizes (dx=dy)
dx = (x_max-x_min)/Nx
dy = (y_max - y_min)/Ny

#cells and  faces             
xf = np.linspace (-1,1,Nx+1)
xc = 0.5*(xf[1:] + xf[:-1])
yf =  np.linspace(-0.5,0.5,Ny+1)
yc = 0.5*(yf[1:] + yf[:-1])

XF,YF = np.meshgrid(xf,yf)
XC,YC = np.meshgrid(xc,yc)

#============================
#FLUID  AND  SOLID Parameters
#============================

#Mean  velocity of parabolic profile
U = 1

#=============================================================================================
#Thermal parameters: (k- #thermal conductivity  ; alpha - thermal diffusivity ; rho - density)
#=============================================================================================
#fluid-thermal parameters 
k_f = 1.0                  
alpha_f = 0.03               
rhoc_f = k_f / alpha_f       

#solid - thermal parameters
k_s = 11.0                 
alpha_s = 0.30              
rhoc_s = k_s / alpha_s      

#Inlet temperature
T_in = 300                  

#Heat flux parameters
q0 = 50            #baseline heating          
A = 200             #amplitude        
sigma = 0.1         #std  deviation          
x0 = -0.9           #initial mean          
V = 0.3         #velocity of initial  mean    

#=============================
#Masks
#=============================

#bounds (masks)
is_bot = yc<-0.3
is_fluid = np.abs(yc)<(0.3)
is_top = yc>0.3

#=============================
#INITIALIZATION
#=============================
#row-wise values:
k_c_row = np.where(is_fluid, k_f,k_s)
k_f_row = 2*(k_c_row[1:]*k_c_row[:-1])/(k_c_row[1:]+k_c_row[:-1])
rhoc_c_row = np.where(is_fluid,rhoc_f,rhoc_s)
u_c_row = np.where(is_fluid, 1.5*U*(1 - (yc/0.3)**2), 0.0)

# #reshaped  matrices (for transpose access in numpy)
k_f_row = k_f_row.reshape(-1,1)
k_c_row = k_c_row.reshape(-1,1)
rhoc_c_row = rhoc_c_row.reshape(-1,1)
u_c_row = u_c_row.reshape(-1,1)

fluid_row_indices = np.linspace((Ny//5),((Ny//5)*4)-1,(Ny//5)*3).astype(int)
solid_top_row_indices = np.linspace(0,(Ny//5)-1,(Ny//5)).astype(int)
solid_bottom_row_indices = np.linspace((Ny//5)*4,Ny-1,(Ny//5)).astype(int)

T_init_cell= np.full((Ny,Nx),T_in)

#==================================
#FVM Function
#==================================
def FVM(T,t,energy=False):
    T_hface_pos = np.zeros((Ny,Nx))
    T_hface_pos_grad = np.zeros_like(T_hface_pos)

    T_hface_neg = np.zeros((Ny,Nx))
    T_hface_neg_grad = np.zeros_like(T_hface_neg)

    F_hface_pos = np.zeros_like(T_hface_pos)
    F_hface_neg = np.zeros_like(T_hface_pos)    

    T_vface_pos_grad = np.zeros((Ny,Nx))
    T_vface_neg_grad = np.zeros_like(T_vface_pos_grad)

    G_vface_pos = np.zeros_like(T_vface_pos_grad)
    G_vface_neg = np.zeros_like(T_vface_pos_grad)

    x_spot = x0 + V*t
    q_flux = np.zeros(Nx)
    for j in range(Nx):
     q_flux[j] = q0 + A*np.exp(-(xc[j] - x_spot)**2 / (2*sigma**2))


    def Fluid():
    # 1) Fluxes:
     for i in fluid_row_indices:
      for j in range(Nx):


    # a) Positive  Horizontal fluxes:
        #interior rows
        if (j+1)<=Nx-1:
         T_hface_pos[i,j] = (T[i,j+1]+T[i,j])/2
         T_hface_pos_grad[i,j] = (T[i,j+1]-T[i,j])/dx
         F_hface_pos[i,j] = rhoc_c_row[i,0]*u_c_row[i,0]*T_hface_pos[i,j] - k_c_row[i,0]*T_hface_pos_grad[i,j]
        #interface rows
        else:
          T_hface_pos[i,j] =  T[i,j]
          T_hface_pos_grad[i,j]=0.0
          F_hface_pos[i,j] = rhoc_c_row[i,0]*u_c_row[i,0]*T_hface_pos[i,j]

    #b) negative Horizontal fluxes:
        #interior rows
        if (j-1)>=0:

         T_hface_neg[i,j] = (T[i,j]+T[i,j-1])/2  
         T_hface_neg_grad[i,j] = (T[i,j]-T[i,j-1])/dx
         F_hface_neg[i,j] = rhoc_c_row[i,0]*u_c_row[i,0]*T_hface_neg[i,j] - k_c_row[i,0]*T_hface_neg_grad[i,j]    
        #interface rows
        else:
           T_hface_neg[i,j] = T_in
           T_hface_neg_grad[i,j] = (T[i,j] - T_in)/(dx/2)
           F_hface_neg[i,j] = rhoc_c_row[i,0]*u_c_row[i,0]*T_in - k_c_row[i,0]*T_hface_neg_grad[i,j]  

    #c) Positive vertical fluxes:
      # interior rows
      for j in range(Nx):

        if  (i-1)>=(Ny//5)+1:
         T_vface_pos_grad[i,j] = (T[i-1,j]-T[i,j])/dy #(+ is - here )
         G_vface_pos[i,j] = -k_f_row[i-1,0]*T_vface_pos_grad[i,j]

      #interface rows
        else:
         T_vface_pos_grad[i,j] = (T[i-1,j]-T[i,j])/dy
         G_vface_pos[i,j] = -k_f_row[i-1,0]*T_vface_pos_grad[i,j]

    #d) Negative vertical fluxes
        #interior  rows
        if(i+1)<=((Ny//5)*4)-1:
          T_vface_neg_grad[i,j] = (T[i,j]-T[i+1,j])/dy
          G_vface_neg[i,j] = -k_f_row[i,0]*T_vface_neg_grad[i,j]
        #interface rows
        else:
           T_vface_neg_grad[i,j] = (T[i,j]-T[i+1,j])/dy
           G_vface_neg[i,j] = -k_f_row[i,0]*T_vface_neg_grad[i,j]

            
    def TopSolid():
    #1) Top Solid
      for i in solid_top_row_indices:
        for j in range(Nx):
         
    #a) Positive Horizontal fluxes:    
      # interior solid cell rows:
         if(j+1)<=Nx-1:  
          T_hface_pos_grad[i,j] = (T[i,j+1]-T[i,j])/dx
          F_hface_pos[i,j] = -k_c_row[i,0]*T_hface_pos_grad[i,j]
      # interface solid cell rows:    
         else:
           F_hface_pos[i,j] = 0.0

    #b) Negative Horizontal fluxes:
      #interior solid cell rows:
         if(j-1)>=0: 
            T_hface_neg_grad[i,j] = (T[i,j]-T[i,j-1])/dx      
            F_hface_neg[i,j] = -k_c_row[i,0]*T_hface_neg_grad[i,j]
      # top solid cell row
         else:
            F_hface_neg[i,j]=0.0

    #c) positive vertical fluxes
      #interiors:
        for j in range(Nx):
          if (i-1)>=0 and (i-1)<(Ny//5)-1:
            T_vface_pos_grad[i,j]=(T[i-1,j]-T[i,j])/dy
            G_vface_pos[i,j]=-k_f_row[i-1,0]*T_vface_pos_grad[i,j]

          #top solid cell row:
          else:
             G_vface_pos[i,j]=0.0

    #d) negative vertical fluxes
      #interiors:
          if (i+1)<(Ny//5):
            T_vface_neg_grad[i,j] = (T[i,j]-T[i+1,j])/dy
            G_vface_neg[i,j] = -k_f_row[i,0]*T_vface_neg_grad[i,j]

      #interface:
          else:
            T_vface_neg_grad[i,j] = (T[i,j]-T[i+1,j])/dy
            G_vface_neg[i,j] = -k_f_row[i,0]*T_vface_neg_grad[i,j]


    def BottomSolid():
    #a)Positive Horizontal  fluxes
      for i in solid_bottom_row_indices:
        for j in range(Nx):
      #interiors:
         if (j+1)<=Nx-1:
          T_hface_pos_grad[i,j] = (T[i,j+1]-T[i,j])/dx
          F_hface_pos[i,j] = -k_c_row[i,0]*T_hface_pos_grad[i,j]
      #interface
         else:
           F_hface_pos[i,j] = 0.0

    #b) negative horizontal fluxes:
      #interiors:
         if (j-1)>=0:
           T_hface_neg_grad[i,j] = (T[i,j]-T[i,j-1])/dx
           F_hface_neg[i,j]= -k_c_row[i,0]*T_hface_neg_grad[i,j]
      #interface:
         else:
           F_hface_neg[i,j]=0.0  

    #c) positive vertical fluxes:
        #interiors:
         if (i-1)>=(Ny//5)+((Ny//5)*3) and (i-1)<Ny-1:
           T_vface_pos_grad[i,j]=(T[i-1,j]-T[i,j])/dy
           G_vface_pos[i,j] = -k_f_row[i-1,0]*T_vface_pos_grad[i,j]

        #interface:
         else:
           T_vface_pos_grad[i,j] = (T[i-1,j]-T[i,j])/dy 
           G_vface_pos[i,j] = -k_f_row[i-1,0]*T_vface_pos_grad[i,j]

    #d) negative vertical fluxes:
         if (i+1)>(Ny//5)+((Ny//5)*3) and  (i+1)<=Ny-1:
          T_vface_neg_grad[i,j] = (T[i,j]-T[i+1,j])/dy
          G_vface_neg[i,j] = -k_f_row[i,0]*T_vface_neg_grad[i,j]

         else:
          G_vface_neg[i,j] = q_flux[j]

    Fluid()
    TopSolid()
    BottomSolid()  
  
    F_sum = (F_hface_pos-F_hface_neg)/dx + (G_vface_pos-G_vface_neg)/dy
    
    f=fluid_row_indices

    F_in=np.zeros(Nx-1);F_out=np.zeros(Nx-1)

    F_in = rhoc_c_row[f,0]*u_c_row[f,0]*T_in - k_c_row[f,0]*((T[f,0]-T_in)/(dx/2))
    F_out = rhoc_c_row[f,0]*u_c_row[f,0]*T[f,Nx-1]
    P_n = np.sum(q_flux)*dx + np.sum(F_in)*dy - np.sum(F_out)*dy

    if not energy:
     return -(F_sum/(rhoc_c_row))
    else:
      return -(F_sum/(rhoc_c_row)), P_n

#=========================================
#TIMESTEP Controls and STABILITY Limits
#=========================================
dt_diff = dx**2/(4*alpha_s)
dt_adv = dx/np.max(u_c_row)
dt = 0.5* np.minimum(dt_adv,dt_diff)
Pe_c = np.max(u_c_row)*dx/alpha_f
Pe = U*(2)/alpha_f
#set_trace()
nt=30000
t=0.0

#====================================
#SNAPSHOT STORAGE AND INTIALIZATIONS
#====================================
T_old=T_init_cell
T_new = np.zeros_like(T_old)
T_hist=[T_old]
P_n_hist=[]

#=======================================
#SOLVER LOOP
#======================================
if SOLVE:
    for it in range(1,nt):
        print("it: ",it)

        Flux_derivative, P_n = FVM(T_old,t,energy=True)
        T_new = T_old+dt*Flux_derivative

        if (it+1)%100==0:
         T_hist.append(T_new)
         
        T_old=T_new.copy()

        P_n_hist.append(P_n)
        
        t+=dt
        
    def EnergyBalance(P_n_hist,T_hist,dt):
        E_0 = np.sum(rhoc_c_row*T_init_cell)*dx*dy
        E_n = np.sum(rhoc_c_row*T_hist[-1])*dx*dy
        dE = E_n-E_0
        Q_net = np.sum(P_n_hist)*dt
        print(f"dE={dE}, Q_net={Q_net}, absolute relative error (%) ={(np.abs(dE-Q_net))*100/(np.abs(Q_net))}")

    EnergyBalance(P_n_hist,T_hist,dt)
    

if PLOT:
  def Plot(T_hist,dt):
    T_min=np.min(T_hist)
    T_max=np.max(T_hist)

    fig,ax=plt.subplots(1,figsize=(10,8))
    m = ax.pcolormesh(XF,YF,T_hist[0],edgecolor='None',shading = 'flat',cmap = 'inferno',vmin=T_min,vmax=T_max)
    ax.axhline(y_int,color='w',linewidth=1.2)
    ax.axhline(-y_int,color='w',linewidth=1.2)
    title=ax.set_title('')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    ax.axhline(0.3,color='white',linewidth=1.0)
    ax.axhline(-0.3,color='white',linewidth=1.0)
    cb=fig.colorbar(m,ax=ax,label='Temp(K)',shrink=0.5)
    cb.set_ticks(np.linspace(T_min,T_max,8))
    plt.tight_layout()

    def update(n):
        t=n*100*dt
        m.set_array(T_hist[n].ravel())
        title.set_text(f"Unsteady CHT - Temperature Distribution (t={t:.3f}s)")
        return m,title

    return FuncAnimation(fig,update,frames=len(T_hist),interval=50)

  ani=Plot(T_hist,dt)
  ani.save('CHT.mp4',writer ='ffmpeg',fps=23)
  plt.show()




















