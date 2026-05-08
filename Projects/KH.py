import numpy as np
import cupy as cp 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
# from IPython.display import HTML
# from google.colab import files

plt.style.use('dark_background')

# ── Parameters
Lx, Ly  = 8.0, 3.0
Nx, Ny  =  1024, 384
Re      = 100000
Pe      = 200000
dx      = Lx / Nx
dy      = Ly / Ny
dt      = 0.0003
nsteps  = 40000
nsave   = 20

# ── Grid
x  = cp.linspace(0, Lx, Nx, endpoint=False)
y  = cp.linspace(0, Ly, Ny, endpoint=False)
X, Y = cp.meshgrid(x, y)

# ── Wavenumbers
kx      = cp.fft.fftfreq(Nx, d=dx / (2 * cp.pi))
ky      = cp.fft.fftfreq(Ny, d=dy / (2 * cp.pi))
KX, KY  = cp.meshgrid(kx, ky)
K2      = KX**2 + KY**2
K2[0,0] = 1.0

# ── Dealiasing mask
dealias = cp.ones((Ny, Nx), dtype=cp.float64)
dealias[Ny//3 : 2*Ny//3, :] = 0.0
dealias[:, Nx//3 : 2*Nx//3] = 0.0

# ── Sponge BC
sponge_width    = 0.35
sponge_strength = 60.0
bot = sponge_width * Ly
top = (1.0 - sponge_width) * Ly

sigma = cp.where(Y < bot, sponge_strength * ((bot - Y) / bot)**2, cp.where(Y > top, sponge_strength * ((Y - top) / (Ly - top))**2, 0.0))

hard_mask = (Y < 0.12 * Ly) | (Y > 0.88 * Ly)

# Initial conditions
delta   = 0.07
epsilon = 0.02
U_inf   = 1.0

U = U_inf + cp.tanh((Y - Ly / 2) / delta)

cp.random.seed(42)
noise     = cp.random.randn(Ny, Nx)
noise_hat = cp.fft.fft2(noise) * dealias
noise     = cp.real(cp.fft.ifft2(noise_hat))
noise    /= cp.std(noise)

dU_dy  = (cp.roll(U, -1, axis=0) - cp.roll(U, 1, axis=0)) / (2 * dy)
omega  = -dU_dy + epsilon * noise
omega0 = -dU_dy.copy()

omega0 = cp.where(hard_mask, 0.0, omega0)   #NOTE:eliminate boundary spikeS in omega0

phi  = 0.5 * (1.0 + cp.tanh((Y - Ly / 2) / delta))
phi0 = phi.copy()

# Layer 0: two-fluid phi
cmap_phi = LinearSegmentedColormap.from_list(
    'two_fluid_bright',
    [(0.00, '#ff0000'),
     (0.40, '#ff4400'),
     (0.50, "#5EFF00FF"),
     (0.60, '#0044ff'),
     (1.00, '#0000ff')],
    N=512
)

# Layer 1: omega

cmap_omega = LinearSegmentedColormap.from_list(
    'silver_glitter',
    [(0.00, "#ffffff"),   # pure white — negative vorticity
     (0.35, '#888888'),   # gray
     (0.50, '#000000'),   # black — zero
     (0.65, '#888888'),   # gray
     (1.00, '#ffffff')],  # pure white — positive vorticity
    N=512
)


def solve_poisson(omega):
    psi_hat       = cp.fft.fft2(omega) / K2
    psi_hat[0, 0] = 0.0
    return cp.real(cp.fft.ifft2(psi_hat))

def get_velocity(psi):
    u =  (cp.roll(psi, -1, axis=0) - cp.roll(psi, 1, axis=0)) / (2 * dy)
    v = -(cp.roll(psi, -1, axis=1) - cp.roll(psi, 1, axis=1)) / (2 * dx)
    return u, v


def rhs_omega(omega):
    psi  = solve_poisson(omega)
    u, v = get_velocity(psi)

    domega_dx = (cp.roll(omega, -1, axis=1) - cp.roll(omega, 1, axis=1)) / (2 * dx)
    domega_dy = (cp.roll(omega, -1, axis=0) - cp.roll(omega, 1, axis=0)) / (2 * dy)

    lap       = (
        (cp.roll(omega, -1, axis=1) - 2*omega + cp.roll(omega, 1, axis=1)) / dx**2 +
        (cp.roll(omega, -1, axis=0) - 2*omega + cp.roll(omega, 1, axis=0)) / dy**2
    )
    val     = -u*domega_dx - v*domega_dy + (1.0/Re)*lap - sigma*(omega - omega0)
    val_hat = cp.fft.fft2(val) * dealias

    return cp.real(cp.fft.ifft2(val_hat))

def rhs_phi(phi, u, v):
    dphi_dx = (cp.roll(phi, -1, axis=1) - cp.roll(phi, 1, axis=1)) / (2 * dx)
    dphi_dy = (cp.roll(phi, -1, axis=0) - cp.roll(phi, 1, axis=0)) / (2 * dy)

    lap_phi = (
        (cp.roll(phi, -1, axis=1) - 2*phi + cp.roll(phi, 1, axis=1)) / dx**2 +
        (cp.roll(phi, -1, axis=0) - 2*phi + cp.roll(phi, 1, axis=0)) / dy**2
    )

    val =  -u*dphi_dx - v*dphi_dy + (1.0/Pe)*lap_phi - sigma*(phi - phi0)
    val_hat = cp.fft.fft2(val)*dealias

    return (cp.real(cp.fft.ifft2(val_hat)))
    

# Time integration
phi_frames   = []
omega_frames = []
time_vals    = []

for it in range(nsteps):
    k1 = rhs_omega(omega)
    k2 = rhs_omega(omega + 0.5*dt*k1)
    k3 = rhs_omega(omega + 0.5*dt*k2)
    k4 = rhs_omega(omega + dt*k3)
    omega = omega + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    omega = cp.where(hard_mask, 0.0, omega)

    psi  = solve_poisson(omega)
    u, v = get_velocity(psi)

    k1s = rhs_phi(phi, u, v)
    k2s = rhs_phi(phi + 0.5*dt*k1s, u, v)
    k3s = rhs_phi(phi + 0.5*dt*k2s, u, v)
    k4s = rhs_phi(phi + dt*k3s, u, v)
    phi = phi + (dt/6.0) * (k1s + 2*k2s + 2*k3s + k4s)
    phi = cp.where(hard_mask, phi0, phi)
    phi = cp.clip(phi, 0.0, 1.0)

    if it % nsave == 0:
        phi_frames.append(phi.get())
        omega_frames.append(omega.get())
        time_vals.append(it * dt)
        if it % 500 == 0:
            print(f"it: {it:5d}/{nsteps},  t={it*dt:.3f},  max|ω|={float(cp.max(cp.abs(omega))):.4f}")


print("Rendering Visual")

lim = float(np.percentile(np.abs(omega_frames[-1]), 97))

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlabel('x', color='white', fontweight='bold')
ax.set_ylabel('y', color='white', fontweight='bold')
ax.tick_params(colors='white', labelsize=10, width=1.5, length=5)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')
for sp in ax.spines.values():
    sp.set_edgecolor('#444444')
    sp.set_linewidth(1.0)

im_phi = ax.imshow(
    phi_frames[0],
    origin='lower',
    extent=[0, Lx, 0, Ly],
    cmap=cmap_phi,
    vmin=0, vmax=1,
    interpolation='antialiased',
    aspect='auto',
    zorder=0,
)

im_omega = ax.imshow(
    omega_frames[0],
    origin='lower',
    extent=[0, Lx, 0, Ly],
    cmap=cmap_omega,
    vmin=-lim, vmax=lim,
    interpolation='antialiased',
    aspect='auto',
    alpha=0,
    zorder=1,
)

ax.set_title(
    'Kelvin–Helmholtz Instability  |  Re = 80,000',
    fontsize=13, color='#cccccc', fontweight='bold'
)

time_text = ax.text(
    0.99, 0.97,
    't = 0.000\nCreated By: Anirudh Renganathan, 2026',
    transform=ax.transAxes,
    fontsize=8, fontweight='bold', color='white',
    ha='right', va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='#444444', alpha=0.7),
    zorder=2,
)

def animate(i):
    im_phi.set_data(phi_frames[i])
    im_omega.set_data(omega_frames[i])
    time_text.set_text(f't = {time_vals[i]:.3f}\nCreated By: Anirudh Renganathan, 2026')
    return im_phi, im_omega, time_text

ani = animation.FuncAnimation(
    fig, animate,
    frames=len(phi_frames),
    interval=30,
    blit=True,
    repeat=True,
)

plt.tight_layout()
# plt.show()

writer = animation.FFMpegWriter(fps=30, bitrate=3000)
ani.save('kh_animation.mp4', writer=writer, dpi=150)
print("saved")


# from IPython.display import HTML
# plt.close()
# HTML(ani.to_jshtml())

# writer = animation.FFMpegWriter(fps=30, bitrate=3000)
# ani.save('kh_animation.mp4', writer=writer, dpi=150)
# files.download('kh_animation.mp4')