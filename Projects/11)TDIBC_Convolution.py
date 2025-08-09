'''SIMULATION OF AN ACOUSTIC PRESSURE WAVE WITH THE IMPLEMENTATION OF A TIME-DOMAIN IMPEDANCE BOUNDARY CONDITION ON THE LEFT WALL
USING COMPACT FINITE DIFFERENCE SCHEME AND MACCORMACK TIME INTEGRATION - COMPRESSIBLE FLOWS CFD'''

#Import Built in functions from python modules, load libraries
import numpy as np 
from numpy import pi as PI
import pylab as plt 
import scipy
from scipy import sparse
from numpy import fft
from scipy.linalg import circulant
from scipy.sparse.linalg import spsolve
from sys import getsizeof  # ACCESS SYSTEM TOOLS
from pdb import set_trace # DEBUG MODE
from numpy.fft import fft
from scipy.sparse.linalg import factorized
import matplotlib.animation as animation
from math import sqrt
from numpy.fft import fft
# from scipy.linalg import toeplitz
# from scipy.linalg import circulant

#Set Grid size:
grid_sizes = [(3*10**i) +1 for i in range(2,3)]  # [10, 100, 1000, 10000]
rms_errors_array=[]

for grid in grid_sizes:

 # N-1 segments, N faces
 Ndft=grid-1

# ASCII art:
#                        x=0......                                        x=L
# finite-volume Grid:     | -o- | -o- | -o- | -o- | -o- | -o- | -o- | -o- |     (9 Faces, 8 cells)
# Working Grid:           | -o- | -o- | -o- | -o- | -o- | -o- | -o- | -o-       (8 faces, 8 cells, Neglect last face)

#BUT THE LAST CELL IS A GHOST CELL - Used only to inteprolate the interior face values, So:
# EFFECTIVE FLOW DOMAIN:  | -o- | -o- | -o- | -o- | -o- | -o- | -o- |            (8 Faces, 7 cells)


 Nseg=Ndft
 x0 = -1.0
 Lx = +1.0  # last grid point, x_{N-1}
 dx = (Lx-x0) / Nseg  # uniform grid
 dx2 = dx * dx
 xf = np.linspace(-1.0, Lx, grid, endpoint=True)  # linearly spaced grid
 xc = 0.5 * (xf[1:] + xf[:-1])  # no repeated periodic points

# BASE STATE:
 rho0  = 1.2 # kg/m^3
 T0    = 300.0 # Kelvin
 P0    = 101000.0 # Atmospheric

# calculate gas constant R
 R_gas = P0/rho0/T0 # J/kg/K
 gamma = 1.4
 a0 = np.sqrt(gamma*R_gas*T0)

#Wave parameters
 Amp= 1e-7#1e-05 #5e-01
 k_wave = 4.0 # 1/length
 sigma = 25.0
 phi_x= 0.2/k_wave

#Initial Wave function
 def initial_waveform(XX):
     return np.exp(-sigma*XX*XX)*np.cos(2.0*PI*k_wave*(XX-phi_x))
    
 p_fluct = Amp*rho0*a0*a0*initial_waveform(xc) # Pascals
 p_fluct_0=Amp*rho0*a0*a0*initial_waveform(xf)

# assume isentropic flow, ds = 0
    
 rho_fluct = p_fluct/a0/a0 # isentropic flow relation

# EULER EQUATION -- INITIAL CONDITION SETUP
 P_init      = P0+p_fluct # total instantaneous pressure (initial + fluctuating)
 rho_init    = rho0+rho_fluct
 T_init   = P_init/rho_init/R_gas
 u_init = p_fluct/rho0/a0 # this guarantees only right-traveling waves, u' = p'/rho0/a0 (but -p --->  left moving)
 Cv = R_gas/(gamma-1.0)
 e_init = Cv*T_init
 E_init = rho_init*e_init + 0.5*rho_init*u_init*u_init

 rho_old, u_old, T_old, P_old = rho_init, u_init, T_init, P_init


 #TIME STEPPING: ---> used to get 'dt' for CONVOLUTION
 t_final=(Lx-x0)/a0
 CFL = 0.1
 ref_Mach = 1.5
 ref_wave_speed = a0 * ref_Mach
 plot_solution = True
 time = 0
 dt = CFL * dx / ref_wave_speed
 nt=8*int(np.ceil(t_final/dt))

#TDIBC Setup:

#IMPEDANCE PARAMETERS:
 R = 8909
 X1 = 0.001842
 X2 = 9703.2390
 w0 = sqrt(float(X1/X2))
 Z0 = rho0*a0 
 
 #Z_w = Z0*(R+(1j*(w*X1 - X2/w)))
 #S_w = 2*Z0/(Z0+Z_w)

 a = 542.9859
 b = 124.5988
 c = -513.3750
 d = 2237.2233
 C = 0
 tau = np.arange(0,(8*int(np.ceil(t_final/dt)))*dt,dt)
 S_tau = np.e**(c*tau) * (2*a*np.cos(d*tau)+((2*(a*c)-C)/d)*np.sin(d*tau))

 #CONVOLUTION PARAMETERS:
 conv_result=0  #convolution result
 history=0   #count
 N_history=len(tau)
 v_out=np.zeros(N_history)   #EMPTY 1D List to store the values of Left-wall's v_out

#Wall BCs: Exrapolate EXTREMUM By setting boundary interpolation to ZERO --> ADIABATIC WALL
 T_old[0] = -((47/8)*(T_old[1]) - (31/8)*(T_old[2]) + (23/24)*(T_old[3])) / (-71/24)   # Zero Temperature gradient at left wall
 T_old[-2] = -((47/8)*(T_old[-3]) - (31/8)*(T_old[-4]) + (23/24)*(T_old[-5])) / (-71/24)  # Zero Temperature gradient at right wall

#IMPOSE THE TDIBC AT THE LEFT WALL:
 v_in_left=0
 v_in_current = v_in_left
 v_out_left=u_old[0]-((P_old[0] - P0)/(rho0*a0))
 v_out_current=v_out_left
 v_out[history]=v_out_current
#  history=(history+1) % N_history

#EVALUATE THE CONVOLUTION INTEGRAL:
 for i in range(N_history):
      tau_idx = (history - i) % N_history #Circular access (1,0,8998,8997,....,2)
      conv_result += S_tau[i] * v_out[tau_idx] * dt
    
 v_in_current = -v_out_current + conv_result

 u_old[0]=0.5*(v_in_current+v_out_current)

 P_old[0]=P0+0.5*rho0*a0*(v_in_current-v_out_current)

 #NO TDIBC AT RIGHT WALL. JUST HARD REFLECTION (No-slip, adiabatic wall)
 v_in_right=u_old[-2]+((P_old[-2] - P0)/(rho0*a0))
 v_out_right=v_in_right

 u_old[-2]=0.5*(v_in_right+v_out_right)

 P_old[-2]=P0+0.5*rho0*a0*(v_in_right-v_out_right)

 #Update e and E --> linked to Temperature T:
 e_init = Cv*T_init
 E_init = rho_init*e_init + 0.5*rho_init*u_init*u_init

 e_old, E_old=e_init, E_init

 ## CONSERVATIVE VARIABLES (Q0,Q1,Q2)
 Q0_init=rho_old            # for mass equation
 Q1_init=rho_old*u_old     # for momentum equation
 Q2_init=E_old             # for energy equation

 Q0_old, Q1_old, Q2_old = Q0_init, Q1_init, Q2_init
 
 #INITIAL FLUXES AT CC:
 massflux_old=Q1_old
 momflux_old = Q1_old * Q1_old / Q0_old + P_old
 energyflux_old = u_old * (Q2_old + P_old)                                                

#Rewrite the 3 fluxes in terms of Q0,Q1,Q2 for convenience in computation:
 flux_Q0_old = Q1_old
 flux_Q1_old = (3 - gamma)/2 * Q1_old**2 / Q0_old + (gamma - 1) * Q2_old
 flux_Q2_old = gamma * Q1_old * Q2_old / Q0_old - (gamma - 1)/2 * Q1_old**3 / Q0_old**2
                                                            

#INTERPOATION FUNCTION: (interpolates function values from Cells to Faces)

 def matfunc1(Y):    # INTERPOLATES FACE-CENTERED FLUXES, FROM CELL-CENTERED FLUXES
    N = len(Y)
    A = 1 * np.eye(N) + (1/6)*np.eye(N, k=1) + (1/6)*np.eye(N, k=-1)  #USE ALPHA=1/6 - 4TH ORDER Accurate ; and consequently b=0
    A[0, 1] = A[-1,-2] = 0 #ALPHA=0 for Boundary FACES
    # A_sparse = sparse.csr_matrix(A)
    # solve_A = factorized(A_sparse)

    B_rhs = np.zeros(N) 
    for i in range(N):
        if i==0:
         term0= ((15/8) * (Y[i])) -((5/4) * (Y[i+1])) + ((3/8) *(Y[i+2]))
         B_rhs[i] =  term0

        if i==N-1:
         term_last= ((15/8) * (Y[i-1])) -((5/4) * (Y[i-2])) + ((3/8) * (Y[i-3]))
         B_rhs[i] =  term_last

        elif i in range(1,N-1):
         term1 = (4/(3*2)) * (Y[i] + Y[i - 1])  # f_{i+1/2} + f_{i-1/2}
         #term2 = (1/(8*2)) * (Y[i + 1] + Y[i - 2])  # f_{i+3/2} + f_{i-3/2}
         B_rhs[i] =  term1 #+ term2  
    return np.linalg.solve(A,B_rhs)
 
#(FACES TO CELL-CENTERS derivative operator)
 def matfunc2(Z): # GIVES DERIVATIVES ON THE CELL-CENTERS FROM FACE-FUNCTION VALUES
    M=len(Z)
    P = 1 * np.eye(M) + (1/22)*np.eye(M, k=1) + (1/22)*np.eye(M, k=-1) #USE ALPHA=1/22 - 4TH ORDER Accurate ; and consequently b=0
    P[0,1]=P[-1,-2]=0 #ALPHA=0 for Boundary FACES
    # P_sparse=sparse.csr_matrix(P)
    # solve_P = factorized(P_sparse)
     
    N_rhs=np.zeros(M)

    for i in range(M):
      if i==0:
       term0= (-(23/24)*(Z[i]) + (7/8)*(Z[i+1]) +(1/8)*(Z[i+2]) - (1/24)*(Z[i+3]))/dx
       N_rhs[i] =  term0

      if i==M-1:
       term_last= (-(23/24)*(Z[i-1]) + (7/8)*(Z[i-2]) +(1/8)*(Z[i-3]) - (1/24)*(Z[i-4]))/dx
       N_rhs[i] =  term_last

      elif i in range(1,M-1):
       term1 = (12 /(11*dx)) * (Z[i+1]-Z[i])
       #term2 = (19 /((24*3*dx))) * (Z[i+2]-Z[i-1])
       N_rhs[i] =  term1 #+ term2  
    return np.linalg.solve(P,N_rhs) 
 
 #Cell Centers to Face-derivative Operator  -> USED FOR CHECKING IF dT/dx=0 at WALLS --> Adiabatic Wall
 def matfunc3(Z):  
    M=len(Z)
    P = 1 * np.eye(M) + (1/22)*np.eye(M, k=1) + (1/22)*np.eye(M, k=-1)  #USE ALPHA=1/22 - 4TH ORDER Accurate ; and consequently b=0
    P[0,1]=P[-1,-2]=0 #ALPHA=0 for Boundary FACES
    # P_sparse=sparse.csr_matrix(P)
    # solve_P=factorized(P_sparse)
     
    N_rhs=np.zeros(M)

    for i in range(M):
      if i==0:
       term0= (-(71/24)*(Z[i]) + (47/8)*(Z[i+1]) -(31/8)*(Z[i+2]) + (23/24)*(Z[i+3]))/dx
       N_rhs[i] =  term0

      if i==M-1:
       term_last= (-(71/24)*(Z[i-1]) + (47/8)*(Z[i-2]) -(31/8)*(Z[i-3]) + (23/24)*(Z[i-4]))/dx
       N_rhs[i] =  term_last

      elif i in range(1,M-1):
       term1 = (12 /(11*dx)) * (Z[i+1]-Z[i])
       #term2 = (19 /((24*3*dx))) * (Z[i+2]-Z[i-1])
       N_rhs[i] =  term1 #+ term2  
    return np.linalg.solve(P,N_rhs) 
 
#INITIAL FUNCTION CALLS OUTSIDE TIME LOOP TO SET INITIAL CONDITIONS AND INITIAL BCs" 

 flux0_old_L=matfunc1(flux_Q0_old)
 flux1_old_L=matfunc1(flux_Q1_old)
 flux2_old_L=matfunc1(flux_Q2_old)

 #CHECKS:
 P_interp1=matfunc1(P_old)
 u_interp=matfunc1(u_old) #-----> should be zero for the last face (wall)
 T_inter=matfunc1(T_old)
 T_der1=matfunc3(T_old) #-----> should be zero

 #TIME STEPPING: (Same as prev. Used To check before the time loop)
 t_final=(Lx-x0)/a0
 CFL = 0.1
 ref_Mach = 1.5
 ref_wave_speed = a0 * ref_Mach
 plot_solution = True
 time = 0
 dt = CFL * dx / ref_wave_speed
 nt= 8*int(np.ceil(t_final/dt))

#INITIALIZE 1-D Lists and arrays for storing iterables, set initial conditions to 0:
 pressure_history=[]
 time_history=[]
 u_history=[]

 v_out_star=np.zeros(N_history)
 conv_result_star=0
 conv_result=0

#  set_trace()

#TIME INTEGRATION:
 for it in range(nt):
    print("it= ",it)
    if it % 10 == 0:
      pressure_history.append(P_old - P0)
      time_history.append(it*dt)
      u_history.append(u_old)

    Q0_star = Q0_old - (dt) * matfunc2(flux0_old_L)
    Q1_star = Q1_old - (dt) * matfunc2(flux1_old_L)
    Q2_star = Q2_old - (dt) * matfunc2(flux2_old_L)

    rho_star = Q0_star
    u_star = Q1_star / Q0_star
    kinetic_energy = 0.5 * rho_star * u_star * u_star
    P_star = (gamma - 1.0) * (Q2_star - kinetic_energy)
    T_star = P_star / R_gas / rho_star
    e_star = Cv*T_star
    E_star= rho_star*e_star+ 0.5*rho_star*u_star*u_star
    
    # WALL BCs:
    T_star[0] = -((47/8)*(T_star[1]) - (31/8)*(T_star[2]) + (23/24)*(T_star[3])) / (-71/24)   # Zero Temperature gradient at left wall
    T_star[-2] = -((47/8)*(T_star[-3]) - (31/8)*(T_star[-4]) + (23/24)*(T_star[-5])) / (-71/24)  # Zero Temperature gradient at right wall

   #AT LEFT WALL:
    v_in_left_star=0
    v_in_current_star = v_in_left_star
    v_out_left_star=u_star[0]-((P_star[0] - P0)/(rho0*a0))
    v_out_current_star=v_out_left_star
    v_out_star[history]=v_out_current_star
   #  history=(history+1) % N_history

    for i in range(N_history):
      tau_idx = (history - i) % N_history  # Circular buffer access
      conv_result += S_tau[i] * v_out_star[tau_idx] * dt
    
    v_in_current_star = -v_out_current_star + conv_result_star

    u_star[0]=0.5*(v_in_current+v_out_current)
    P_star[0]=P0+0.5*rho0*a0*(v_in_current-v_out_current)

    #AT RIGHT WALL:
    v_in_right_star=u_star[-2]+((P_star[-2] - P0)/(rho0*a0))
    v_out_right_star=v_in_right_star

    u_star[-2]=0.5*(v_in_right_star+v_out_right_star)
    P_star[-2]=P0+0.5*rho0*a0*(v_in_right_star-v_out_right_star)

    Q0_star=rho_star           
    Q1_star=rho_star*u_star     
    Q2_star=E_star              

    flux_Q0_star = Q1_star
    flux_Q1_star = ((3 - gamma)/2) * Q1_star**2 / Q0_star + (gamma - 1) * Q2_star
    flux_Q2_star = gamma * Q1_star * Q2_star / Q0_star - ((gamma - 1)/2) * Q1_star**3 / Q0_star**2

    P_interp2=matfunc1(P_star)

    flux0_star_L = matfunc1(flux_Q0_star) 
    flux1_star_L = matfunc1(flux_Q1_star)
    flux2_star_L = matfunc1(flux_Q2_star)

    T_der2=matfunc3(T_star)

    Q0_new = 0.5 * (Q0_old + Q0_star) - 0.5 * (dt) * matfunc2(flux0_star_L)
    Q1_new = 0.5 * (Q1_old + Q1_star) - 0.5 * (dt) * matfunc2(flux1_star_L)
    Q2_new = 0.5 * (Q2_old + Q2_star) - 0.5 * (dt) * matfunc2(flux2_star_L)

    Q0_old = Q0_new.copy()
    Q1_old = Q1_new.copy()
    Q2_old = Q2_new.copy()

    rho_new = Q0_old
    u_new= Q1_old / Q0_old
    kinetic_energy = 0.5 * rho_new * u_new * u_new
    P_new = (gamma - 1.0) * (Q2_old - kinetic_energy)
    T_new = P_new / R_gas / rho_new
    e_new = Cv*T_new
    E_new= rho_new*e_new+ 0.5*rho_new*u_new*u_new

    rho_old, u_old, T_old, P_old, e_old, E_old = rho_new, u_new, T_new, P_new, e_new, E_new

    #WALL BCs
    T_old[0] = -((47/8)*(T_old[1]) - (31/8)*(T_old[2]) + (23/24)*(T_old[3])) / (-71/24)   # Zero Temperature gradient at left wall
    T_old[-2] = -((47/8)*(T_old[-3]) - (31/8)*(T_old[-4]) + (23/24)*(T_old[-5])) / (-71/24)  # Zero Temperature gradient at right wall

   #AT LEFT WALL:
    v_in_left=0
    v_in_current = v_in_left
    v_out_left=u_old[0]-((P_old[0] - P0)/(rho0*a0))
    v_out_current=v_out_left
    v_out[history]=v_out_current
   #  history=(history+1) % N_history

    for i in range(N_history):
      tau_idx = (history - i) % N_history  # Circular access
      conv_result += S_tau[i] * v_out[tau_idx] * dt
    
    v_in_current = -v_out_current + conv_result

    u_old[0]=0.5*(v_in_current+v_out_current)

    P_old[0]=P0+0.5*rho0*a0*(v_in_current-v_out_current)

    #AT RIGHT WALL:
    v_in_right=u_old[-2]+((P_old[-2] - P0)/(rho0*a0))
    v_out_right=v_in_right

    u_old[-2]=0.5*(v_in_right+v_out_right)

    P_old[-2]=P0+0.5*rho0*a0*(v_in_right-v_out_right)

    rho_old, u_old, T_old, P_old = rho_new, u_new, T_new, P_new

    Q0_old=rho_old          
    Q1_old=rho_old*u_old     
    Q2_old=E_old             

    flux_Q0_old = Q1_old
    flux_Q1_old = ((3 - gamma)/2) * Q1_old**2 / Q0_old + (gamma - 1) * Q2_old
    flux_Q2_old = gamma * Q1_old * Q2_old / Q0_old - ((gamma - 1)/2) * Q1_old**3 / Q0_old**2

    flux0_old_L=matfunc1(flux_Q0_old)
    flux1_old_L=matfunc1(flux_Q1_old) 
    flux2_old_L=matfunc1(flux_Q2_old) 

    rho_old, u_old, T_old, P_old = rho_new, u_new, T_new, P_new

    conv_result=0
    conv_result_star=0
    history=(history+1) % N_history
   
    time+=dt

#CALCULATE FLUCTIATING PRESSURE:
 P_Euler=P_new-P0

#FINAL CHECKS:
 u_interp_end=matfunc1(u_old)
 P_interp_end=matfunc1(P_old)
 T_inter_end=matfunc1(T_old)
 T_der_end=matfunc3(T_old)


#PLOT-RELATED ADJUSTMENTS:
 x_shifted = (xc - a0 * time - x0) % (Lx - x0) + x0
 p_fluct_exact = Amp * rho0 * a0**2 * initial_waveform(x_shifted)

 xc_plot = xc[0:-1]
 #set_trace()

#PLOTTING:
 if plot_solution:
    figwidth = 16
    figheight = 8
    lineWidth = 3
    textFontSize = 30
    gcafontSize = 20

    # Create figure for animation
    fig = plt.figure(0, figsize=(figwidth, figheight))
    ax = fig.add_subplot(1, 1, 1)
    
    # Initialize lines with your exact formatting
    #init_line, = ax.plot(xc[0:-1], p_fluct[0:-1], ':', linewidth=2, label="Initial condition")
    #euler_line, = ax.plot(xc[0:-1], pressure_history[0][0:-1], '-r', linewidth=7, label="Euler Equations")
    u_line, = ax.plot(xc[0:-1],u_history[0][0:-1], '-b', linewidth=7, label="Velocity Wave")
    #vin_line, = ax.plot(xf[0:-2], vin_history[0][0:-1],'-b', linewidth=7, label="V_in")
    #vout_line, = ax.plot(xc[0:-1], vout_history[0][0:-1],'-k', linewidth=7, label="V_out")
    
    ax.set_title(f"Pade 4th Order interpolation-MacCormack, A = {Amp}, T={0:.5f}s", fontsize=textFontSize)
    #ax.set_ylim([-3 * Amp * rho0 * a0 * a0, 3 * Amp * rho0 * a0 * a0])
    ax.set_ylim([-5e-5,+5e-5])
    plt.setp(ax.get_xticklabels(), fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(), fontsize=gcafontSize)
    ax.grid('on', which='both')
    ax.set_xlabel("$x$", fontsize=textFontSize)
    ax.set_ylabel("$p'$ (Pa)", fontsize=textFontSize, rotation=90)
    plt.legend(fontsize=textFontSize)
    plt.tight_layout()

    # Animation update function
    def update(frame):
        # Update only the Euler line data
        #euler_line.set_ydata(pressure_history[frame][0:-1])
        #vin_line.set_ydata(vin_history[frame][0:-1])
        #vout_line.set_ydata(vout_history[frame][0:-1])
        u_line.set_ydata(u_history[frame][0:-1])
        ax.set_title(f"Pade 4th Order interpolation-MacCormack, A = {Amp},T={round(float(dt * nt), 5)}s", 
                    fontsize=textFontSize)
        #return euler_line,
        #return vin_line,
        #return vout_line,
        return u_line,

    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(pressure_history),
        interval=50,  # m-s between frames
        blit=True
    )

    plt.show()

    # Save final frame as PDF
    plt.savefig('euler_solution_visible.pdf')
