import numpy as np
from scipy.fft import fft2, ifft2
from mpi4py import MPI
from parallel_fft import parallel_fft2, parallel_ifft2
from utils import compute_velocity_from_vorticity

class NavierStokesSolver:
    def __init__(self, grid_size, viscosity, dt, forcing_amplitude):
        # Inicializamos el comunicador MPI y obtenemos info del proceso
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Parametros de la simulacion
        self.N = grid_size              # Tamaño de la malla (N x N)
        self.nu = viscosity             # Viscosidad del fluido
        self.dt = dt                    # Paso de tiempo
        self.A = forcing_amplitude      # Amplitud del termino de forzamiento
        
        # Checamos si hay mas procesos que celdas, lo cual no tiene sentido
        if self.size > self.N:
            if self.rank == 0:
                print(f"Error: Numero de procesos ({self.size}) es mayor que el tamaño de la malla ({self.N}).")
            self.comm.Abort(1)  # Cancelamos la simulación
        
        # Calculamos el tamaño local de la malla que le toca a este proceso
        self.N_local = self.N // self.size
        if self.rank < self.N % self.size:
            self.N_local += 1  # Distribuimos lo que sobra entre los primeros procesos
        
        # Creamos la malla espacial y los numeros de onda para la FFT
        self.x = np.linspace(0, 1, self.N, endpoint=False)  # Malla de 0 a 1 (sin incluir 1)
        self.dx = 1.0 / self.N                              # Espaciado entre puntos
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)  # Numeros de onda para FFT (frecuencias en el espacio)
        
        # Creamos las mallas locales de numeros de onda (solo las que le tocan a este proceso)
        if self.N_local > 0:
            self.kx_local, self.ky = np.meshgrid(
                self.k, self.k[self.rank * self.N_local:(self.rank + 1) * self.N_local], indexing='ij'
            )
        else:
            self.kx_local, self.ky = np.zeros((self.N, 0)), np.zeros((self.N, 0))  # Caso sin celdas locales
        
        # Inicializamos el campo de vorticidad con un vórtice gaussiano
        self.w = np.zeros((self.N, self.N_local), dtype=np.complex128)  # Campo complejo por la FFT
        if self.rank == 0:
            X, Y = np.meshgrid(self.x, self.x)
            self.w = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))  # Vortice en el centro
        self.w = self.comm.bcast(self.w, root=0)  # Todos los procesos reciben el mismo campo inicial
        
        # Precalculamos el Laplaciano en espacio de Fourier
        self.k2 = self.kx_local**2 + self.ky**2
        if self.N_local > 0:
            self.k2[0, 0] = 1.0  # Evitamos division por cero en el modo (0,0)
        
        # Mensaje de depuración
        if self.rank == 0:
            print(f"Rank {self.rank}: w.shape={self.w.shape}, kx_local.shape={self.kx_local.shape}, k2.shape={self.k2.shape}")
    
    def step(self):
        # Por si a este proceso no le toco ninguna fila
        if self.N_local == 0:
            return np.zeros((self.N, 0))
        
        # Calculamos el campo de velocidades (u, v) a partir de la vorticidad
        u, v = compute_velocity_from_vorticity(self.w, self.kx_local, self.ky, self.N)
        
        # Transformamos la vorticidad al espacio de Fourier (fft2 paralela)
        w_hat = parallel_fft2(self.w, self.comm)
        
        # Calculamos el termino no lineal (convectivo)
        u_grad_w = self.compute_nonlinear_term(u, v, w_hat)
        
        # Calculamos el término viscoso usando el Laplaciano en Fourier
        laplacian_w = -self.k2 * w_hat
        
        # Aplicamos el forzamiento externo en x (tipo seno)
        if self.rank == 0:
            forcing = self.A * np.sin(2 * np.pi * self.x)[:, None]  # Solo en x
        else:
            forcing = np.zeros((self.N, self.N_local))
        forcing_hat = parallel_fft2(forcing, self.comm)
        
        # Paso de tiempo explícito tipo Euler hacia adelante
        dw_dt = -u_grad_w + self.nu * laplacian_w + forcing_hat  # Ecuación principal
        w_hat += self.dt * dw_dt  # Avanzamos la solución
        self.w = parallel_ifft2(w_hat, self.comm)  # Volvemos al espacio real
        
        return self.w.real  # Regresamos la parte real del campo
    
    def compute_nonlinear_term(self, u, v, w_hat):
        # Si este proceso no tiene datos, no calcula nada
        if self.N_local == 0:
            return np.zeros((self.N, 0), dtype=np.complex128)
        
        # Convertimos vorticidad de Fourier al espacio real para derivadas
        w = parallel_ifft2(w_hat, self.comm).real
        dw_dx = parallel_ifft2(1j * self.kx_local * w_hat, self.comm).real  # Derivada parcial en x
        dw_dy = parallel_ifft2(1j * self.ky * w_hat, self.comm).real       # Derivada parcial en y
        
        # Termino no lineal: u * (dw/dx) + v * (dw/dy)
        nonlinear = u * dw_dx + v * dw_dy
        
        # Transformamos el termino de vuelta a Fourier
        return parallel_fft2(nonlinear, self.comm)