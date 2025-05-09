import numpy as np
from mpi4py import MPI
from parallel_fft import parallel_fft2, parallel_ifft2

def compute_velocity_from_vorticity(w, kx, ky, N):
    # Iniciamos el comunicador MPI y obtenemos el rank del proceso actual
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Si la grilla local esta vacía, regresamos arrays vacios pa u y v
    if w.shape[1] == 0:
        return np.zeros((N, 0)), np.zeros((N, 0))
    
    # Debug: mostramos las formas (shapes) de los datos si somos el proceso 0
    if rank == 0:
        print(f"compute_velocity: w.shape={w.shape}, kx.shape={kx.shape}, ky.shape={ky.shape}")
    
    # Calculamos la vortocidad (w) en el espacio de Fourier usando fft 2D paralela
    w_hat = parallel_fft2(w, comm)
    
    # Calculamos la función corriente (stream function): psi_hat = -w_hat / k^2
    k2 = kx**2 + ky**2  # módulo al cuadrado del vector de onda
    if w.shape[1] > 0:
        k2[0, 0] = 1.0  # evitamos dividir entre cero en el modo (0,0)
    psi_hat = -w_hat / k2  # ecuacion de Poisson en espacio de Fourier
    
    # Calculamos la velocidad a partir de la función corriente:
    # u = d(psi)/dy  → derivada respecto a y, se multiplica por iky en Fourier
    # v = -d(psi)/dx → derivada respecto a x, se multiplica por -ikx en Fourier
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    
    # Transformamos de regreso al espacio físico con fft inversa
    u = parallel_ifft2(u_hat, comm).real  # Nos quedamos solo con la parte real
    v = parallel_ifft2(v_hat, comm).real
    
    # Retornamos los campos de velocidad u y v
    return u, v
