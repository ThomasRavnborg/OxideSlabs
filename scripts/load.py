import time
from ase.io import iread

def test_load_time(file_path, type='extended xyz'):
    t0 = time.time()
    for i, structure in enumerate(iread(file_path, index=':')):
        if i == 0:
            print('ASE atoms object:')
            print(structure)
            try:
                print('Forces:')
                print(structure.get_forces())
            except:
                print("Could not retrieve forces.")
            try:
                print('Velocities:')
                print(structure.get_velocities())
            except:
                print("Could not retrieve velocities.")
    t1 = time.time()
    print(f"Time taken to read {type} file: {t1 - t0:.2f} seconds")

test_load_time('results/ALnep/iteration_2/nve_production/Ba8O24Ti8/movie/dump.xyz',
                type='extended xyz')
test_load_time('results/ALnep/iteration_2/nve_production/Ba8O24Ti8/exxyz/movie.xyz',
                type='movie xyz')
test_load_time('results/ALnep/iteration_2/nve_production_test2/Ba8O24Ti8/500K/movie.nc',
                type='netcdf')