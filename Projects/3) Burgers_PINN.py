import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy import pi as PI
from numpy import fft as fft

# ==============================================================
# 1. Solution to the Burgers'  Equation (non conservative form)
# ==============================================================
N=200+1 # N-1 segments, N points
Ndft=N-1
Nseg=Ndft

#GEOMETRY SETTINGS
x0=0.0
Lx=2*PI # this is the last grid point, x_{N-1}
dx = Lx/Nseg # uniform grid for now
dx2=dx*dx
xf = np.linspace(0,Lx,N)
xc = 0.5*(xf[1:]+xf[:-1])

#Initialize a base acoustic pulse
base_vel=1.0
uu0 = 3.0*np.sin(2.0*PI/Lx*2.0*xc+PI/3.0)+base_vel
characteristic_velocity = base_vel
max_vel = np.max(uu0)

# SIMULATION PARAMETERS
nt=1900
CFL=0.01
dt=CFL*dx/max_vel
t_array  = np.linspace(0,nt*dt,nt)

u_old_LO = uu0

dft_dealiasing = True

u_new_LO = np.zeros_like(uu0)

u_hist=[]

#Build  k_array
k_arr = np.hstack([np.arange(0,Ndft//2+1),np.arange(-Ndft//2+1,0)])
Ndft_ext = int((5/2)*Ndft)
k_arr_ext = np.hstack([np.arange(0,Ndft_ext//2+1),np.arange(-Ndft_ext//2+1,0)])

#ITERATOR LOOP : (Need to DEALIAS - Need to kill the high-frequency modes to  avoid aliasing errors)
for it in range(nt):

    if dft_dealiasing:

        U_hat_old = fft.fft(u_old_LO)/Ndft

        V_hat_arr = 1j*np.zeros(Ndft_ext)

        V_hat_arr[0:Ndft//2+1]=U_hat_old[0:Ndft//2+1]

        V_hat_arr[-Ndft//2+1:]=U_hat_old[-Ndft//2+1:]

        vv_old = fft.ifft(V_hat_arr*Ndft_ext)

        vv_old_sq = np.real(vv_old*vv_old)

        vv_old_sq_hat = fft.fft(vv_old_sq)/Ndft_ext

        vv_old_sq_hat[Ndft//2]=0.0

        U2_old_hat = np.delete(vv_old_sq_hat,np.arange(Ndft//2+1,Ndft_ext-Ndft//2+1))

        uu2_old_hat = U2_old_hat*Ndft

        flux_hat_uu_old = 0.5*(uu2_old_hat)

    else:

        flux_hat_uu_old=(fft.fft(0.5*(u_old_LO)**2))


    #Forward diff - Euler Explicit (RK-1)
    du_dx = fft.ifft(2*PI*1j*k_arr/Lx * flux_hat_uu_old)
    u_new_LO = u_old_LO-dt*(np.real(du_dx))
    u_old_LO = u_new_LO

    u_hist.append(u_old_LO)

#Write final value of u_hist to file
# u_hist = np.array(u_hist)
# with open ("u_hist_final.txt","w") as  f:
#     for i in  range(len(u_hist[-1])):
#         f.write(str(u_hist[-1][i])+"\n")

# Need to  pair  xc and   time array
X,T=np.meshgrid(xc,t_array)

#Convert data to tensor for PINN
x_data = torch.tensor(X.flatten(),dtype=torch.float32).reshape(-1,1)
t_data = torch.tensor(T.flatten(),dtype=torch.float32).reshape(-1,1)
y_data = torch.tensor(u_hist.flatten(),dtype=torch.float32).reshape(-1,1)

# ==========================================
# 2. PINN DEFINITION
# ==========================================
class PINN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layer1 = nn.Linear(size, 16)
        self.relu = nn.Tanh()
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 16)
        self.layer6 = nn.Linear(16, 1)

    def forward(self, xt):
        x = self.layer1(xt)  
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.relu(x)
        x = self.layer6(x)
        return x

model = PINN(2)
opti = optim.Adam(model.parameters(), lr=0.0009)

print("Training PINN")
# ==========================================
# 3. HYBRID TRAINING LOOP
# ==========================================
epoch = 10000
batch  = 20000

for ep in range(epoch):
    indices = torch.randperm(len(x_data))[:batch]

    x_train = x_data[indices].clone().requires_grad_(True)
    t_train = t_data[indices].clone().requires_grad_(True)
    y_data_clone = y_data[indices].clone()
    xt_train = torch.cat([x_train,t_train],dim=1)

    y_pred=model(xt_train)

    data_loss = torch.mean((y_pred-y_data_clone)**2)

    dy_dt_pred = torch.autograd.grad(y_pred,t_train,grad_outputs=torch.ones_like(y_pred),create_graph=True)[0]
    dy_dx_pred = torch.autograd.grad(y_pred,x_train,grad_outputs=torch.ones_like(y_pred),create_graph=True)[0]

    physics_loss = torch.mean(((y_pred*dy_dx_pred)+(dy_dt_pred))**2)

    loss = physics_loss+data_loss

    loss.backward()

    opti.step()
    opti.zero_grad()

    if ep%500==0:
        print(f"epoch - {ep}: loss-{loss.item()}")

#Save the model
torch.save(model.state_dict(),'burgers_pinn.pth')

#===========================================================
# Loading and testing the Trained Model on the final state
#===========================================================

#Load the final numerical solution:
# u_final=[]
# with  open  ("u_hist_final.txt","r") as  f:
#     for line in  f:
#         s=line.strip()
#         u_final.append(float(s))        

loaded_model = PINN(2)

loaded_model.load_state_dict(torch.load('burgers_pinn.pth'))
loaded_model.eval()

x_test = torch.tensor(xc,dtype=torch.float32,requires_grad=True).reshape(-1,1)
t_test = torch.full_like(x_test, t_array[-1]).reshape(-1,1)
xt_test = torch.cat([x_test,t_test],dim=1)

with torch.no_grad():
    y_predi = loaded_model(xt_test).numpy()

#PLOT SOLUTIONS
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(xc, uu0, 'r--', linewidth=3, label="IC (Initial Condition)")
ax.plot(xc, u_hist[-1], 'b-', linewidth=4, label='PINN Surrogate')
ax.plot(xc, y_predi, 'ko', markersize=10, markerfacecolor='None',markeredgewidth=2, label='RK-1 Euler Explicit')    
ax.set_xlabel('x', fontsize=18, fontweight='bold')
ax.set_ylabel('u(x,t)', fontsize=18, fontweight='bold')
ax.set_title("u(x,t) vs x - Solution of Burgers'Equation (Pseudo-Spectral)", fontsize=20, fontweight='bold', pad=15)
ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=2.5)
plt.setp(ax.get_xticklabels(), fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')

for spine in ax.spines.values():
    spine.set_linewidth(2.5)

t_final = t_array[-1]
amp = 3.0  
legend_info = f"Time: t = {t_final:.3f}s \nAmplitude: {amp}"

ax.legend(title=legend_info, title_fontsize=15, fontsize=14, framealpha=1.0, edgecolor='black')
ax.get_legend().get_title().set_fontweight('bold')
ax.grid(True, which='both', linestyle='--', linewidth=1.5, alpha=0.6)

plt.tight_layout()
plt.show()



      
