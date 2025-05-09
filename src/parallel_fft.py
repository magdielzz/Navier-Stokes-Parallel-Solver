import numpy as np
from mpi4py import MPI
from scipy.fft import fft2, ifft2

def parallel_fft2(data, comm):
    # Obtener info del proceso actual y del total de procesos
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = data.shape[0]  # tamaño global del grid
    N_local = data.shape[1] if len(data.shape) > 1 else 0  # tamaño local del pedazo del grid

    # Mensaje de depuración para ver los tamaños y procesos
    if rank == 0:
        print(f"parallel_fft2: N={N}, N_local={N_local}, size={size}, data.shape={data.shape}")
    
    # Si el grid local está vacío, regresamos un arreglo vacío
    if N_local == 0:
        return np.zeros((N, 0), dtype=np.complex128)
    
    # Aplicamos la FFT solo en la dirección x (columas), esto se puede hacer localmente en cada proceso
    data_hat = fft2(data, axes=(0,), norm='ortho')

    # Reunimos el número de elementos que cada proceso va a mandar
    local_sendcount = data_hat.size
    sendcounts = np.zeros(size, dtype=int)
    comm.Allgather([np.array(local_sendcount, dtype=int), MPI.INT], [sendcounts, MPI.INT])

    # Calculamos los desplazamientos para saber donde va cada parte del arreglo global
    displacements = np.zeros(size, dtype=int)
    if rank == 0:
        displacements[1:] = np.cumsum(sendcounts[:-1])
    comm.Bcast(displacements, root=0)

    # Debug para ver cómo se va a juntar la info de todos los procesos
    if rank == 0:
        print(f"parallel_fft2: sendcounts={sendcounts}, displacements={displacements}")
    
    # Creamos un arreglo vacio para juntar toda la data fft desde todos los procesos
    data_hat_global = np.zeros((N, N), dtype=np.complex128)
    try:
        # Todos los procesos mandan su parte al arreglo global
        comm.Allgatherv(data_hat, [data_hat_global, sendcounts, displacements, MPI.DOUBLE_COMPLEX])
    except Exception as e:
        # Si hay un error lo imprimimos y abortamos
        if rank == 0:
            print(f"Error en Allgatherv: {e}")
        comm.Abort(1)
    
    # Ahora hacemos la FFT en la dirección y (filas), ya que ya tenemos toda la data global
    data_hat_global = fft2(data_hat_global, axes=(1,), norm='ortho')

    # Creamos un arreglo para guardar la parte local otra vez después de la fft global
    data_hat_local = np.zeros((N, N_local), dtype=np.complex128)
    for i in range(size):
        start = displacements[i] // N
        end = start + (sendcounts[i] // N)
        if rank == i:
            data_hat_local[:, :] = data_hat_global[:, start:end]
    
    return data_hat_local

def parallel_ifft2(data_hat, comm):
    # Obtener info del proceso actual
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = data_hat.shape[0]
    N_local = data_hat.shape[1] if len(data_hat.shape) > 1 else 0

    # Mensaje de depuración para ver tamaños
    if rank == 0:
        print(f"parallel_ifft2: N={N}, N_local={N_local}, size={size}, data_hat.shape={data_hat.shape}")
    
    # Si no hay grid local, regresamos vacio
    if N_local == 0:
        return np.zeros((N, 0), dtype=np.complex128)
    
    # Creamos espacio para juntar toda la data
    data_global = np.zeros((N, N), dtype=np.complex128)
    local_sendcount = data_hat.size
    sendcounts = np.zeros(size, dtype=int)
    comm.Allgather([np.array(local_sendcount, dtype=int), MPI.INT], [sendcounts, MPI.INT])

    displacements = np.zeros(size, dtype=int)
    if rank == 0:
        displacements[1:] = np.cumsum(sendcounts[:-1])
    comm.Bcast(displacements, root=0)

    # Juntamos toda la data de todos los procesos
    try:
        comm.Allgatherv(data_hat, [data_global, sendcounts, displacements, MPI.DOUBLE_COMPLEX])
    except Exception as e:
        if rank == 0:
            print(f"Error en Allgatherv: {e}")
        comm.Abort(1)

    # Hacemos la FFT inversa en la dirección y (filas)
    data_global = ifft2(data_global, axes=(1,), norm='ortho')

    # Regresamos cada pedazo local a su proceso correspondiente
    data_local = np.zeros((N, N_local), dtype=np.complex128)
    for i in range(size):
        start = displacements[i] // N
        end = start + (sendcounts[i] // N)
        if rank == i:
            data_local[:, :] = data_global[:, start:end]
    
    # Ahora aplicamos la FFT inversa en x (columnas), localmente
    data_local = ifft2(data_local, axes=(0,), norm='ortho')

    return data_local
