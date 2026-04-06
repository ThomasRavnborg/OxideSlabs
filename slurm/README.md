Slurm batch scripts.
Submit jobs with
sbsubmit <script.py> <mode>
where <mode> can be choosen among the following:
- SIESTA:   Used to run SIESTA calculation scripts on CPUs
- GPAW:     Used to run GPAW calculation scripts on CPUs
- A100:     Used to run calculations on GPUs when submitting code from surt.fysik.dtu.dk
- H200:     Used to run calculations on GPUs when submitting code from sara.fysik.dtu.dk