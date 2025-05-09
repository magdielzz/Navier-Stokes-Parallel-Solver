Navier-Stokes Parallel Solver
Project Summary
This project implements a parallel solver for the 2D Navier-Stokes equations in vorticity form using a pseudo-spectral method. The simulation evolves a vorticity field on a periodic domain, initialized with a Gaussian vortex, and includes a small viscosity and a sinusoidal forcing term. The solver parallelizes computations across multiple processes using MPI, with FFTs distributed via the mpi4py library. The final vorticity field is visualized as a contour plot.
How It Works

Physics: The Navier-Stokes equations are solved in vorticity form:[ \frac{\partial \omega}{\partial t} + \mathbf{u} \cdot \nabla \omega = \nu \nabla^2 \omega + f ] where ( \omega ) is vorticity, ( \mathbf{u} = (u, v) ) is the velocity field, ( \nu ) is viscosity, and ( f ) is a forcing term.
Numerical Method: Uses a pseudo-spectral method:
Spatial derivatives are computed in Fourier space using FFTs.
Time-stepping is performed with forward Euler.


Parallelization:
The grid is partitioned across processes, with each process handling a subset of columns.
FFTs are parallelized using a two-stage approach: local FFTs along one axis, followed by global FFTs after data redistribution via MPI.


Output: The vorticity field at the final time is gathered to the root process and plotted as a contour map (vorticity.png).

Files

main.py: Entry point; sets up parameters, runs the simulation, and generates the plot.
src/navier_stokes_solver.py: Defines the NavierStokesSolver class for time-stepping and core computations.
src/parallel_fft.py: Implements parallel FFT and inverse FFT functions.
src/utils.py: Utility function to compute velocity from vorticity.
run.sh: Bash script to execute the program with MPI.
requirements.txt: Lists required Python packages.

Requirements

System: Ubuntu (tested on WSL on Windows).
Software:
Python 3.6+
MPI implementation (MPICH recommended for WSL; OpenMPI may also work)


Python Packages (listed in requirements.txt):
numpy>=1.21.0
scipy>=1.7.0
mpi4py>=3.1.0
matplotlib>=3.4.0



Setup Instructions
1. Clone or Create the Project Directory
mkdir ~/navier_stokes_parallel_solver
cd ~/navier_stokes_parallel_solver
mkdir src

2. Save the Project Files

Place the following files in ~/navier_stokes_parallel_solver:
main.py
run.sh
requirements.txt
This README.md


Place the following files in ~/navier_stokes_parallel_solver/src:
navier_stokes_solver.py
parallel_fft.py
utils.py



Note: Ensure the contents of these files match the versions provided in the project documentation. They are configured to work together without errors.
3. Install Dependencies
Install the required system packages and Python dependencies.
System Packages (Ubuntu/WSL)
sudo apt update
sudo apt install -y mpich libmpich-dev

Python Packages
pip3 install --user -r requirements.txt

4. Make run.sh Executable
chmod +x run.sh

5. Run the Simulation
Execute the simulation with 2 processes:
bash run.sh

6. View the Output
The simulation will generate vorticity.png in the project directory. If running in WSL, copy the file to Windows to view it:
cp vorticity.png /mnt/c/Users/YourWindowsUsername/Desktop/

Open vorticity.png to see the vorticity field at the final time.
Expected Output

Console Output: Progress messages, including:
Starting simulation with 2 processes
Solver initialized: grid_size=128, N_local=64, w.shape=(128, 64)
Step 0/100, Time: 0.000, w.shape=(128, 64)
Step 100/100, Time: 0.100, w.shape=(128, 64)
Simulation completed, vorticity.png generated


Plot: vorticity.png shows a Gaussian vortex centered at (0.5, 0.5) with peak vorticity ~0.96, slightly diffused due to viscosity.


Troubleshooting

MPI Errors:

If you encounter MPI-related errors, try reinstalling mpi4py:
pip3 uninstall -y mpi4py
pip3 install --user --force-reinstall mpi4py


Alternatively, use OpenMPI instead of MPICH:
sudo apt remove --purge mpich libmpich-dev
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
pip3 install --user --force-reinstall mpi4py




WSL Issues:

Increase WSL resources by editing /mnt/c/Users/YourWindowsUsername/.wslconfig:
[wsl2]
memory=4GB
processors=4

Restart WSL:
wsl --shutdown




No Plot Generated:

Run in single-process mode to debug:
python3 main.py


Check for matplotlib errors and ensure it’s installed.




Customization

Simulation Parameters (in main.py):

grid_size: Increase to 256 for higher resolution.
T: Increase to 1.0 to run longer.
forcing_amplitude: Increase to 1.0 for stronger forcing.


Initial Condition (in navier_stokes_solver.py):

Add multiple vortices:
self.w = (np.exp(-50 * ((X - 0.3)**2 + (Y - 0.3)**2)) +
          np.exp(-50 * ((X - 0.7)**2 + (Y - 0.7)**2)))





