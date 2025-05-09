# Solverdor Paralelo de Navier-Stokes

## Resumen del Proyecto

Este proyecto implementa un solvedor paralelo para las ecuaciones de Navier-Stokes en 2D en forma de vorticidad usando un método pseudo-espectral. La simulación evoluciona un campo de vorticidad en un dominio periódico, inicializado con un vórtice Gaussiano, e incluye una pequeña viscosidad y un término de forzado sinusoidal. El solvedor paraleliza los cálculos entre múltiples procesos usando MPI, con FFTs distribuidas mediante la librería `mpi4py`. El campo final de vorticidad se visualiza como un gráfico de contorno.

### Cómo Funciona

* **Física**: Se resuelven las ecuaciones de Navier-Stokes en forma de vorticidad:
  $\\frac{\\partial \\omega}{\\partial t} + \\mathbf{u} \\cdot \\nabla \\omega = \\nu \\nabla^2 \\omega + f$ donde $\omega$ es la vorticidad, $\mathbf{u} = (u, v)$ es el campo de velocidad, $\nu$ es la viscosidad, y $f$ es un término de forzado.
* **Método Numérico**: Usa un método pseudo-espectral:

  * Las derivadas espaciales se calculan en el espacio de Fourier usando FFTs.
  * El paso de tiempo se realiza con Euler hacia adelante.
* **Paralelización**:

  * La rejilla se particiona entre procesos, con cada proceso manejando un subconjunto de columnas.
  * Las FFTs se paralelizan usando un enfoque en dos etapas: FFTs locales a lo largo de un eje, seguidas por FFTs globales después de redistribuir los datos vía MPI.
* **Salida**: El campo de vorticidad al tiempo final se reune en el proceso raíz y se grafica como un mapa de contornos (`vorticity.png`).

### Archivos

* `main.py`: Punto de entrada; configura parámetros, ejecuta la simulación y genera el gráfico.
* `src/navier_stokes_solver.py`: Define la clase `NavierStokesSolver` para los pasos de tiempo y cálculos principales.
* `src/parallel_fft.py`: Implementa funciones de FFT e inversa de FFT paralelas.
* `src/utils.py`: Función utilitaria para calcular velocidad a partir de la vorticidad.
* `run.sh`: Script bash para ejecutar el programa con MPI.
* `requirements.txt`: Lista de paquetes necesarios en Python.

## Requisitos

* **Sistema**: Ubuntu (probado en WSL sobre Windows).
* **Software**:

  * Python 3.6 o superior
  * Implementación de MPI (MPICH recomendado para WSL; OpenMPI también puede funcionar)
* **Paquetes de Python** (listados en `requirements.txt`):

  * `numpy>=1.21.0`
  * `scipy>=1.7.0`
  * `mpi4py>=3.1.0`
  * `matplotlib>=3.4.0`

## Instrucciones de Configuración

### 1. Clonar o Crear el Directorio del Proyecto

```bash
mkdir ~/navier_stokes_parallel_solver
cd ~/navier_stokes_parallel_solver
mkdir src
```

### 2. Guardar los Archivos del Proyecto

* Coloca los siguientes archivos en `~/navier_stokes_parallel_solver`:

  * `main.py`
  * `run.sh`
  * `requirements.txt`
  * Este `README.md`
* Coloca los siguientes archivos en `~/navier_stokes_parallel_solver/src`:

  * `navier_stokes_solver.py`
  * `parallel_fft.py`
  * `utils.py`

**Nota**: Asegúrate de que el contenido de estos archivos coincida con las versiones proporcionadas en la documentación del proyecto. Están configurados para funcionar juntos sin errores.

### 3. Instalar Dependencias

Instala los paquetes del sistema y dependencias de Python necesarias.

#### Paquetes del Sistema (Ubuntu/WSL)

```bash
sudo apt update
sudo apt install -y mpich libmpich-dev
```

#### Paquetes de Python

```bash
pip3 install --user -r requirements.txt
```

### 4. Hacer Ejecutable `run.sh`

```bash
chmod +x run.sh
```

### 5. Ejecutar la Simulación

Ejecuta la simulación con 2 procesos:

```bash
bash run.sh
```

### 6. Ver la Salida

La simulación generará `vorticity.png` en el directorio del proyecto. Si estás corriendo en WSL, copia el archivo a Windows para verlo:

```bash
cp vorticity.png /mnt/c/Users/TuNombreDeUsuarioWindows/Desktop/
```

Abre `vorticity.png` para ver el campo de vorticidad al tiempo final.

## Salida Esperada

* **Salida en Consola**: Mensajes de progreso, incluyendo:

  ```
  Starting simulation with 2 processes
  Solver initialized: grid_size=128, N_local=64, w.shape=(128, 64)
  Step 0/100, Time: 0.000, w.shape=(128, 64)
  Step 100/100, Time: 0.100, w.shape=(128, 64)
  Simulation completed, vorticity.png generated
  ```

* **Gráfico**: `vorticity.png` muestra un vórtice Gaussiano centrado en (0.5, 0.5) con vorticidad pico de \~0.96, ligeramente difundido por la viscosidad.

## Solución de Problemas

* **Errores de MPI**:

  * Si encuentras errores relacionados con MPI, intenta reinstalar `mpi4py`:

    ```bash
    pip3 uninstall -y mpi4py
    pip3 install --user --force-reinstall mpi4py
    ```

  * Alternativamente, usa OpenMPI en vez de MPICH:

    ```bash
    sudo apt remove --purge mpich libmpich-dev
    sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
    pip3 install --user --force-reinstall mpi4py
    ```

* **Problemas en WSL**:

  * Aumenta los recursos de WSL editando `/mnt/c/Users/TuNombreDeUsuarioWindows/.wslconfig`:

    ```
    [wsl2]
    memory=4GB
    processors=4
    ```

    Reinicia WSL:

    ```bash
    wsl --shutdown
    ```

* **No se Genera el Gráfico**:

  * Ejecuta en modo de un solo proceso para depurar:

    ```bash
    python3 main.py
    ```

  * Revisa si hay errores de `matplotlib` y asegúrate de que esté instalado.

## Personalización

* **Parámetros de Simulación** (en `main.py`):

  * `grid_size`: Aumenta a 256 para mayor resolución.
  * `T`: Aumenta a 1.0 para correr por más tiempo.
  * `forcing_amplitude`: Aumenta a 1.0 para un forzado más fuerte.

* **Condición Inicial** (en `navier_stokes_solver.py`):

  * Agrega múltiples vórtices:

    ```python
    self.w = (np.exp(-50 * ((X - 0.3)**2 + (Y - 0.3)**2)) +
              np.exp(-50 * ((X - 0.7)**2 + (Y - 0.7)**2)))
    ```

---



