import numpy as np # numerical python
from numpy import pi as PI
import pylab as plt # replica of matlab's plotting tools
import scipy
from scipy import sparse
from numpy import fft
from scipy.linalg import circulant
from scipy.sparse.linalg import spsolve
from sys import getsizeof # accesses system tools
from pdb import set_trace # this stops the code and gives you interactive env.
from numpy.fft import fft
from scipy.sparse.linalg import factorized
import matplotlib.animation as animation
from math import sqrt
# from scipy.linalg import toeplitz
# from scipy.linalg import circulant

grid_sizes = [(3*10**i) +1 for i in range(2,3)]  # [10, 100, 1000, 10000]
rms_errors_array=[]


for grid in grid_sizes:

 # N-1 segments, N faces
 Ndft=grid-1

# ASCII art:
#                        x=0......                                        x=L
# finite-volume POV:      | -o- | -o- | -o- | -o- | -o- | -o- | -o- | -o- |
# DFT points (so far):    x --- x --- x --- x --- x --- x --- x --- x ---  
# DFT points (new POV):   | -x- | -x- | -x- | -x- | -x- | -x- | -x- | -x- |

 Nseg=Ndft
 x0 = -1.0
 Lx = +1.0  # last grid point, x_{N-1}
 dx = (Lx-x0) / Nseg  # uniform grid
 dx2 = dx * dx
 xf = np.linspace(-1.0, Lx, grid, endpoint=True)  # linearly spaced grid
 xc = 0.5 * (xf[1:] + xf[:-1])  # no repeated periodic points

# DEFINE A BASE STATE
 rho0  = 1.2 # kg/m^3
 T0    = 300.0 # Kelvin
 P0    = 101000.0 # Atmospheric

# calculate gas constant
 R_gas = P0/rho0/T0 # J/kg/K
 gamma = 1.4
 a0 = np.sqrt(gamma*R_gas*T0)
# R_gas = 280.55555000000 K
# hand clap -> p' ~ 1e-04 Pa

# PI        = 3.141592653589793238462643383279.....
# python PI = 3.14159265358979312
# my speech = 20x1e-06 Pa, base pressure = 10^5 Pa
 Amp= 1e-7#1e-05 #5e-01
 k_wave = 4.0 # 1/length
 sigma = 25.0
 phi_x= 0.2/k_wave

 def initial_waveform(XX):
     return np.exp(-sigma*XX*XX)*np.cos(2.0*PI*k_wave*(XX-phi_x))
    
 p_fluct = Amp*rho0*a0*a0*initial_waveform(xc) # Pascals
 p_fluct_0=Amp*rho0*a0*a0*initial_waveform(xf)
# assume isentropic flow, s' = 0
    
 rho_fluct = p_fluct/a0/a0 # isentropic relation

# EULER EQUATION -- INITIAL CONDITION SETUP
 P_init      = P0+p_fluct # total instantaneous pressure
 rho_init    = rho0+rho_fluct
 T_init   = P_init/rho_init/R_gas
 u_init = p_fluct/rho0/a0 # this guarantees only right-traveling waves, u' = p'/rho0/a0
 Cv = R_gas/(gamma-1.0)
 e_init = Cv*T_init
 E_init = rho_init*e_init + 0.5*rho_init*u_init*u_init

 rho_old, u_old, T_old, P_old = rho_init, u_init, T_init, P_init

#Wall BCs: Exrapolate EXTREMUM By setting boundary interpolation to ZERO
 T_old[0] = -((47/8)*(T_old[1]) - (31/8)*(T_old[2]) + (23/24)*(T_old[3])) / (-71/24)   # Zero Temperature gradient at left wall
 T_old[-2] = -((47/8)*(T_old[-3]) - (31/8)*(T_old[-4]) + (23/24)*(T_old[-5])) / (-71/24)  # Zero Temperature gradient at right wall

 u_old[0] = -(-(5/4)*(u_old[1]) + (3/8)*(u_old[2])) / (15/8)   # No-slip condition at left wall
 u_old[-2] = -(-(5/4)*(u_old[-3]) + (3/8)*(u_old[-4])) / (15/8)  # No-slip condition at right wall

 #Update e and E --> linked to Temperature T:
 e_init = Cv*T_init
 E_init = rho_init*e_init + 0.5*rho_init*u_init*u_init

 e_old, E_old=e_init, E_init

 ## CONSERVATIVE VARIABLES (Q0,Q1,Q2)
 Q0_init=rho_old            # mass, zero-th equation
 Q1_init=rho_old*u_old     # momentum, first equation
 Q2_init=E_old             # energy, second equation

 Q0_old, Q1_old, Q2_old = Q0_init, Q1_init, Q2_init
 
 #INITIAL FLUXES AT CC:
 massflux_old=Q1_old
 momflux_old = Q1_old * Q1_old / Q0_old + P_old
 energyflux_old = u_old * (Q2_old + P_old)                                                

 flux_Q0_old = Q1_old
 flux_Q1_old = (3 - gamma)/2 * Q1_old**2 / Q0_old + (gamma - 1) * Q2_old
 flux_Q2_old = gamma * Q1_old * Q2_old / Q0_old - (gamma - 1)/2 * Q1_old**3 / Q0_old**2
                                                            
 '''#Wall Flux BCs: EXTRAPOLATE THE EXTREMUM by setting the BouNDARY iNTERPOLATION TO ZERO
 flux_Q0_old[0] = -((-(5/4) * flux_Q0_old[1]) + ((3/8) * flux_Q0_old[2])) / (15/8)
 flux_Q0_old[-1] = -((-(5/4) * flux_Q0_old[-2]) + ((3/8) * flux_Q0_old[-3])) / (15/8)
 
 #Momentum Flux should equal the pressure since u=0. So, split the flux into Q terms and P terms, then apply the same extrapolation logic by setting the interpolation to ZERO
 flux_Q1_old[0]= (((15/8) * (P_old[0])) - ((5/4) * (P_old[1])) + ((3/8) *(P_old[2])) -((-(5/4)*(flux_Q1_old[1])) + ((3/8)*(flux_Q1_old[2]))))/(15/8)
 flux_Q1_old[-1]= (((15/8) * (P_old[-1])) - ((5/4) * (P_old[-2])) + ((3/8) *(P_old[-3])) -((-(5/4)*(flux_Q1_old[-2])) + ((3/8)*(flux_Q1_old[-3]))))/(15/8)
 
 flux_Q2_old[0] = -((-(5/4) * flux_Q2_old[1])  + ((3/8) * flux_Q2_old[2])) / (15/8)
 flux_Q2_old[-1] = -((-(5/4) * flux_Q2_old[-2]) + ((3/8) * flux_Q2_old[-3])) / (15/8)'''


# # values INTERPOATION FUNCTION:

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

 def matfunc2(Z): #(FACES TO CELL-CENTERS derivative operator)
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

 def matfunc3(Z):  #(Cell Centers to Face-derivative Operator)  -> USED FOR CHECKING IF dT/dx=0 at WALLS
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
 
 P_interp1=matfunc1(P_old)

 flux0_old_L=matfunc1(flux_Q0_old)
 flux0_old_L[0]=flux0_old_L[-1]=0   # - set to zero

 flux1_old_L=matfunc1(flux_Q1_old) 
 flux1_old_L[0]=P_interp1[0]        # - should use face pressures (extrapolation)
 flux1_old_L[-1]=P_interp1[-1]      # - should use face pressures (extrapolation)

 flux2_old_L=matfunc1(flux_Q2_old)
 flux2_old_L[0]=flux2_old_L[-1]=0   # - set to zero

 u_interp=matfunc1(u_old)
 T_inter=matfunc1(T_old)
 T_der1=matfunc3(T_old)

 #TIME STEPPING:
 t_final=(Lx-x0)/a0
 CFL = 0.1
 ref_Mach = 1.5
 ref_wave_speed = a0 * ref_Mach
 plot_solution = True
 time = 0
 dt = CFL * dx / ref_wave_speed
 nt= 2*int(np.ceil(t_final/dt))

 pressure_history=[]
 time_history=[]

#TIME INTEGRATION:
 for it in range(nt):
    print("it= ",it)
    if it % 10 == 0:
      pressure_history.append(P_old - P0)
      time_history.append(it*dt)

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

    u_star[0] = -(-(5/4)*(u_star[1]) + (3/8)*(u_star[2])) / (15/8)   # No-slip condition at left wall
    u_star[-2] = -(-(5/4)*(u_star[-3]) + (3/8)*(u_star[-4])) / (15/8)  # No-slip condition at right wall

    Q0_star=rho_star           # mass, zero-th equation
    Q1_star=rho_star*u_star     # momentum, first equation
    Q2_star=E_star              # energy, second equation

    flux_Q0_star = Q1_star
    flux_Q1_star = ((3 - gamma)/2) * Q1_star**2 / Q0_star + (gamma - 1) * Q2_star
    flux_Q2_star = gamma * Q1_star * Q2_star / Q0_star - ((gamma - 1)/2) * Q1_star**3 / Q0_star**2

    '''#FLUX WALL BCs:
    flux_Q0_star[0] = -((-(5/4) * flux_Q0_star[1]) + ((3/8) * flux_Q0_star[2])) / (15/8)
    flux_Q0_star[-1] = -((-(5/4) * flux_Q0_star[-2]) + ((3/8) * flux_Q0_star[-3])) / (15/8)

    flux_Q1_star[0] = (((15/8) * (P_star[0])) - ((5/4) * (P_star[1])) + ((3/8) * (P_star[2])) - ((-(5/4) * (flux_Q1_star[1])) + ((3/8) * (flux_Q1_star[2])))) / (15/8)
    flux_Q1_star[-1] = (((15/8) * (P_star[-1])) - ((5/4) * (P_star[-2])) + ((3/8) * (P_star[-3])) - ((-(5/4) * (flux_Q1_star[-2])) + ((3/8) * (flux_Q1_star[-3])))) / (15/8)

    flux_Q2_star[0] = -((-(5/4) * flux_Q2_star[1]) + ((3/8) * flux_Q2_star[2])) / (15/8)
    flux_Q2_star[-1] = -((-(5/4) * flux_Q2_star[-2]) + ((3/8) * flux_Q2_star[-3])) / (15/8)'''

    P_interp2=matfunc1(P_star)

    flux0_star_L = matfunc1(flux_Q0_star) #same as prev
    flux0_star_L[0]=flux0_star_L[-1]=0

    flux1_star_L = matfunc1(flux_Q1_star)
    flux1_star_L[0]=P_interp2[0]
    flux1_star_L[-1]=P_interp2[-1]

    flux2_star_L = matfunc1(flux_Q2_star)
    flux2_star_L[0]=flux2_star_L[-1]=0

    T_der2=matfunc3(T_star)
    T_der2[0]=T_der2[-1]=0

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

    u_old[0] = -(-(5/4)*(u_old[1]) + (3/8)*(u_old[2])) / (15/8)   # No-slip condition at left wall
    u_old[-2] = -(-(5/4)*(u_old[-3]) + (3/8)*(u_old[-4])) / (15/8)  # No-slip condition at right wall

    rho_old, u_old, T_old, P_old = rho_new, u_new, T_new, P_new

    Q0_old=rho_old          # mass, zero-th equation
    Q1_old=rho_old*u_old     # momentum, first equation
    Q2_old=E_old              # energy, second equation

    flux_Q0_old = Q1_old
    flux_Q1_old = ((3 - gamma)/2) * Q1_old**2 / Q0_old + (gamma - 1) * Q2_old
    flux_Q2_old = gamma * Q1_old * Q2_old / Q0_old - ((gamma - 1)/2) * Q1_old**3 / Q0_old**2

    '''#FLUX WALL BCs:
    flux_Q0_old[0] = -((-(5/4) * flux_Q0_old[1]) + ((3/8) * flux_Q0_old[2])) / (15/8)
    flux_Q0_old[-1] = -((-(5/4) * flux_Q0_old[-2]) + ((3/8) * flux_Q0_old[-3])) / (15/8)
 
    flux_Q1_old[0]= (((15/8) * (P_old[0])) - ((5/4) * (P_old[1])) + ((3/8) *(P_old[2])) -((-(5/4)*(flux_Q1_old[1])) + ((3/8)*(flux_Q1_old[2]))))/(15/8)
    flux_Q1_old[-1]= (((15/8) * (P_old[-1])) - ((5/4) * (P_old[-2])) + ((3/8) *(P_old[-3])) -((-(5/4)*(flux_Q1_old[-2])) + ((3/8)*(flux_Q1_old[-3]))))/(15/8)
 
    flux_Q2_old[0] = -((-(5/4) * flux_Q2_old[1])  + ((3/8) * flux_Q2_old[2])) / (15/8)
    flux_Q2_old[-1] = -((-(5/4) * flux_Q2_old[-2]) + ((3/8) * flux_Q2_old[-3])) / (15/8)'''

    P_interp3=matfunc1(P_old)

    flux0_old_L=matfunc1(flux_Q0_old) # - set to zero
    flux0_old_L[0]=flux0_old_L[-1]=0

    flux1_old_L=matfunc1(flux_Q1_old) # - should use face pressures (extrapolation)
    flux1_old_L[0]=P_interp3[0]
    flux1_old_L[-1]=P_interp3[-1]

    flux2_old_L=matfunc1(flux_Q2_old) # - set to zero
    flux2_old_L[0]=flux2_old_L[-1]=0


    T_der3=matfunc3(T_old)

    rho_old, u_old, T_old, P_old = rho_new, u_new, T_new, P_new
   
    time+=dt

 P_Euler=P_new-P0

 u_interp_end=matfunc1(u_old)
 P_interp_end=matfunc1(P_old)
 T_inter_end=matfunc1(T_old)
 T_der_end=matfunc3(T_old)
 
 #set_trace()

 x_shifted = (xc - a0 * time - x0) % (Lx - x0) + x0
 p_fluct_exact = Amp * rho0 * a0**2 * initial_waveform(x_shifted)

 xc_plot = xc[0:-1]
 #set_trace()

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
    init_line, = ax.plot(xc[0:-1], p_fluct[0:-1], ':', linewidth=2, label="Initial condition")
    euler_line, = ax.plot(xc[0:-1], pressure_history[0][0:-1], '-r', linewidth=7, label="Euler Equations")
    
    ax.set_title(f"Pade 4th Order interpolation-MacCormack, A = {Amp}, T={0:.5f}s", fontsize=textFontSize)
    ax.set_ylim([-3 * Amp * rho0 * a0 * a0, 3 * Amp * rho0 * a0 * a0])
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
        euler_line.set_ydata(pressure_history[frame][0:-1])
        ax.set_title(f"Pade 4th Order interpolation-MacCormack, A = {Amp},T={round(float(dt * nt), 5)}s", 
                    fontsize=textFontSize)
        return euler_line,

    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(pressure_history),
        interval=50,  # ms between frames
        blit=True
    )

    plt.show()

    # Save final frame as PDF
    plt.savefig('euler_solution_visible.pdf')
    '''
    # Phase analysis (using last frame)
    P_Euler = pressure_history[-1]
    x_shifted = (xc - a0 * time_history[-1] - x0) % (Lx - x0) + x0
    p_fluct_exact = Amp * rho0 * a0**2 * initial_waveform(x_shifted)
    
    euler_hat = fft(P_Euler)
    exact_hat = fft(p_fluct_exact)
    k_mode = int(k_wave * (Lx - x0))
    phase_euler = np.angle(euler_hat[k_mode])
    phase_exact = np.angle(exact_hat[k_mode])

    print(f"Final time: {time_history[-1]:.7f}, Expected t_final: {t_final:.7f}")
    print(f"Phase lag (rad): {phase_euler - phase_exact:.5f}")

    '''