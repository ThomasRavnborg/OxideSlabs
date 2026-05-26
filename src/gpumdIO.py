
import os


def create_run_in(ensemble='npt_ber', dt=1, n_steps=5*1e5, n_dump=1000, T0=300, T1=300, bulk=True):

    n_steps = int(n_steps)
    n_dump = int(n_dump)
    delta_dump = n_steps // n_dump

    # Thermostat and barostat parameters
    T_coup = 100
    p_coup = 1000

    # Pressure control parameters
    p_xx = 0
    p_yy = 0
    p_zz = 0
    C_xx = 150
    C_yy = 150
    C_zz = 150

    # Number of replications
    #N_rep = 10

    if ensemble == 'nve':
        # Reduce number of replications
        N_rep = 5

    if bulk:
        directions = ['tri']
    else:
        directions = ['x', 'y', 'xy']

    direction_p1_p2 = ' '.join([f'{dir} 0 0' for dir in directions])

    if T1 != T0:
        run_in = ""
        n_steps = n_steps // 2
    else:
        run_in = ""
        """
        if bulk:
            run_in = f"replicate {N_rep} {N_rep} {N_rep}"
        else:
            run_in = f"replicate {N_rep} {N_rep} 1"
        """
    
    if ensemble == 'npt_ber':
        pass
    elif ensemble == 'pimd':
        n_beads = 16
        ensemble += f" {n_beads}"

    run_in += f"""
        potential ../../../nep.txt

        velocity {T0}
        time_step {dt}
        """
    
    def _setup_single_run(run_in, ensemble, T0, T1):


        if ensemble == 'nve':
            #delta_dump = 20
            #n_steps = int(1*1e4)
            # equilibration
            print(f'Maximum frequency that can be resolved is {500.0 / (dt * delta_dump)} THz.')
            run_in += f"""
                ensemble nvt_lan {T0} {T1} {T_coup}
                run {n_steps//10}
            """
            # production
            run_in += f"""
                dump_exyz {delta_dump} 1
                ensemble nve
                run {n_steps}
            """
            return run_in

        run_in += f"""
            dump_exyz {delta_dump} 0 1
            dump_thermo {delta_dump}"""

        if ensemble.split('_')[0] == 'nvt':
            run_in += f"""
            ensemble {ensemble} {T0} {T1} {T_coup}"""

        elif ensemble == 'npt_mttk':
            run_in += f"""
            ensemble {ensemble} temp {T0} {T1} tperiod {T_coup} {direction_p1_p2} pperiod {p_coup}"""

        elif ensemble.split('_')[0] == 'npt' or ensemble == 'pimd':
            run_in += f"""
            ensemble {ensemble} {T0} {T1} {T_coup} {p_xx} {p_yy} {p_zz} {C_xx} {C_yy} {C_zz} {p_coup}"""
        
        run_in += f"""
            run {n_steps}
        """
        return run_in

    run_in = _setup_single_run(run_in, ensemble, T0, T1)

    if T1 != T0:
        run_in = _setup_single_run(run_in, ensemble, T1, T0)


    run_in_text = "\n".join(line.strip() for line in run_in.splitlines())

    return run_in_text



def save_run_in(dt, n_steps, n_dump, T0, T1, bulk, dir):

    n_steps = int(n_steps)
    n_dump = int(n_dump)
    delta_dump = n_steps // n_dump

    # Thermostat and barostat parameters
    T_coup = 100
    p_coup = 1000

    if bulk:
        directions = ['tri']
    else:
        directions = ['x', 'y', 'xy']

    direction_p1_p2 = ' '.join([f'{dir} 0 0' for dir in directions])


    if T1 != T0:
        run_in = ""
        n_steps = n_steps // 2
        delta_dump = delta_dump
    else:
        if bulk:
            run_in = "replicate 10 10 10"
        else:
            run_in = "replicate 10 10 1"

    run_in += f"""
        potential ../../../nep.txt

        velocity {T0}
        time_step {dt}

        dump_exyz {delta_dump} 0 1
        dump_thermo {delta_dump}
        ensemble npt_mttk temp {T0} {T1} tperiod {T_coup} {direction_p1_p2} pperiod {p_coup}
        run {n_steps}
    """
    if T1 != T0:
        run_in += f"""
        dump_exyz {delta_dump} 0 1
        dump_thermo {delta_dump}
        ensemble npt_mttk temp {T1} {T0} tperiod {T_coup} {direction_p1_p2} pperiod {p_coup}
        run {n_steps}
        """


    with open(os.path.join(dir, "run.in"), "w") as f:
        text = "\n".join(line.strip() for line in run_in.splitlines())
        f.write(text)


def save_run_in_old(dt, n_steps, n_dump, T, bulk, dir):

    n_steps = int(n_steps)
    n_dump = int(n_dump)
    delta_dump = n_steps // n_dump

    if bulk:
        direction = 'tri 0 0'
    else:
        direction = 'x 0 0 y 0 0 xy 0 0'

    # Create run.in file for GPUMD
    run_in = f"""# --- system ---
        potential ../../../nep.txt

        # --- initialization ---
        velocity {T}
        time_step {dt}

        # --- Stage 1: NVT equilibration ---
        dump_thermo {delta_dump}
        dump_exyz {delta_dump} 0 1
        ensemble nvt_nhc {T} {T} 100
        run {n_steps}

        # --- Stage 2: NPT production ---
        dump_thermo {delta_dump}
        dump_exyz {delta_dump} 0 1
        ensemble npt_mttk temp {T} {T} {direction} tperiod 100 pperiod 1000
        run {n_steps}
    """

    with open(os.path.join(dir, "run.in"), "w") as f:
        text = "\n".join(line.strip() for line in run_in.splitlines())
        f.write(text)
