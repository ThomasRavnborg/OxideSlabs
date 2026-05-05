
import os

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

        # --- output ---
        dump_thermo {delta_dump}
        dump_exyz {delta_dump} 0 1

        # --- md ---
        ensemble npt_mttk temp {T} {T} {direction}
        run {n_steps}
    """

    with open(os.path.join(dir, "run.in"), "w") as f:
        text = "\n".join(line.strip() for line in run_in.splitlines())
        f.write(text)


def save_run_in(dt, n_steps, n_dump, T, bulk, dir):

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
