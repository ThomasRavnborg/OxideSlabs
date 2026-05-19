
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

    if T1 != T0:
        run_in = ""
        n_steps = n_steps // 2
        delta_dump = delta_dump // 2
    else:
        if bulk:
            run_in = "replicate 10 10 10"
        else:
            run_in = "replicate 10 10 1"
    
    if ensemble == 'npt_ber':
        pass
    elif ensemble == 'pimd':
        n_beads = 16
        ensemble += f" {n_beads}"

    run_in += f"""
        potential ../../../nep.txt

        velocity {T0}
        time_step {dt}

        dump_exyz {delta_dump} 0 1
        dump_thermo {delta_dump}
        ensemble {ensemble} {T0} {T1} {T_coup} {p_xx} {p_yy} {p_zz} {C_xx} {C_yy} {C_zz} {p_coup}
        run {n_steps}
        """
    if T1 != T0:
        run_in += f"""
        dump_exyz {delta_dump} 0 1
        dump_thermo {delta_dump}
        ensemble {ensemble} {T1} {T0} {T_coup} {p_xx} {p_yy} {p_zz} {C_xx} {C_yy} {C_zz} {p_coup}
        run {n_steps}
        """
    return run_in



def save_run_in(dt, n_steps, n_dump, T0, T1, bulk, dir):

    n_steps = int(n_steps)
    n_dump = int(n_dump)
    delta_dump = n_steps // n_dump

    if bulk:
        direction = 'tri 0 0'
    else:
        direction = 'x 0 0 y 0 0 xy 0 0'

    if T1 != T0:
        run_in = ""
        n_steps = n_steps // 2
        delta_dump = delta_dump // 20
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
        ensemble npt_mttk temp {T0} {T1} {direction}
        run {n_steps}
    """
    if T1 != T0:
        run_in += f"""
        dump_exyz {delta_dump} 0 1
        dump_thermo {delta_dump}
        ensemble npt_mttk temp {T1} {T0} {direction}
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
