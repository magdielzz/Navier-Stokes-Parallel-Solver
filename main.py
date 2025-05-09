import sys
sys.path.append('src')  # Agregamos la carpeta 'src' al path de Python para poder importar modulos que esten ahi
import numpy as np
from mpi4py import MPI  # Importamos la libreria para hacer computo paralelo con MPI (Message Passing Interface)
from navier_stokes_solver import NavierStokesSolver  # Importamos el solucionador que implementa las ecuaciones de Navier-Stokes
import matplotlib.pyplot as plt  # Para graficar los resultados

def main():
    # Inicializamos el comunicador MPI y obtenemos el rango (rank) del proceso actual
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Solo el proceso con rank 0 imprime el mensaje de inicio para no repetirlo muchas veces
    if rank == 0:
        print(f"Iniciando la simulacion con {comm.Get_size()} procesos")
    
    # Definimos los parametros de la simulacion
    grid_size = 128           # Tamaño de la malla, es decir, cuántos puntos hay en cada dirección (N x N)
    viscosity = 1e-4          # Viscosidad del fluido (nu), un parametro fisico importante
    dt = 0.001                # Paso de tiempo para la integracion temporal
    T = 0.1                   # Tiempo total de simulacion
    forcing_amplitude = 0.1   # Amplitud del término de forzamiento (como si fuera una fuente externa que agita el fluido)
    n_steps = int(T / dt)     # Número total de pasos de tiempo que vamos a dar
    
    # Inicializamos el solucionador de Navier-Stokes con los parámetros definidos
    try:
        solver = NavierStokesSolver(grid_size, viscosity, dt, forcing_amplitude)
        if rank == 0:
            print(f"Solver inicializado: grid_size={grid_size}, N_local={solver.N_local}, w.shape={solver.w.shape}")
    except Exception as e:
        # Si hay un error al crear el solver, se aborta la simulacion
        if rank == 0:
            print(f"Error al inicializar el solver: {e}")
        comm.Abort(1)
    
    # Ejecutamos la simulacion: evolucionamos el campo de vorticidad en el tiempo
    for step in range(n_steps):
        try:
            w = solver.step()  # Avanzamos una paso en el tiempo
            if rank == 0 and step % 100 == 0:
                # Cada 100 pasos se imprime info del progreso
                print(f"Paso {step}/{n_steps}, Tiempo: {step * dt:.3f}, w.shape={w.shape}")
        except Exception as e:
            if rank == 0:
                print(f"Error en el paso {step}: {e}")
            comm.Abort(1)
    
    # Recolectamos el campo de vorticidad de todos los procesos hacia el proceso 0 para graficar
    if rank == 0:
        print("Recolectando resultados para visualizarlos")
    w_global = None
    if rank == 0:
        # w_global almacenara el campo completo (no particionado)
        w_global = np.zeros((grid_size, grid_size), dtype=np.complex128)
    
    # Calculamos cuantos datos envia cada proceso (sendcounts) y donde empiezan (displacements)
    local_sendcount = solver.w.size  # Numero de elementos que tiene cada proceso
    sendcounts = np.zeros(comm.Get_size(), dtype=int)
    comm.Allgather([np.array(local_sendcount, dtype=int), MPI.INT], [sendcounts, MPI.INT])
    
    displacements = np.zeros(comm.Get_size(), dtype=int)
    if rank == 0:
        # Calculamos el desplazamiento de cada bloque de datos en el arreglo global
        displacements[1:] = np.cumsum(sendcounts[:-1])
    comm.Bcast(displacements, root=0)
    
    # Usamos Gatherv para recolectar los datos de cada proceso en el arreglo w_global
    try:
        if rank == 0:
            print(f"Gatherv: sendcounts={sendcounts}, displacements={displacements}, w_global.shape={w_global.shape}, w.shape={solver.w.shape}")
        comm.Gatherv(solver.w, [w_global, sendcounts, displacements, MPI.DOUBLE_COMPLEX], root=0)
    except Exception as e:
        if rank == 0:
            print(f"Error en Gatherv: {e}")
        comm.Abort(1)
    
    # Graficamos el campo de vorticidad al final del tiempo de simulación (solo el rank 0 lo hace)
    if rank == 0:
        print("Graficando resultados")
        plt.figure(figsize=(8, 6))
        # Usamos la parte real de la vorticidad para la grafica
        plt.contourf(solver.x, solver.x, w_global.real.T, levels=50, cmap='viridis')
        plt.colorbar(label='Vorticidad')
        plt.title(f'Vorticidad de Navier-Stokes en t={T:.2f}, nu={viscosity}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('vorticity.png')  # Guardamos la imagen
        plt.close()
        print("Simulacion terminada, imagen vorticity.png generada")

if __name__ == "__main__":
    main()