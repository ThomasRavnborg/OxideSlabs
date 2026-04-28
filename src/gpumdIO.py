
import os

def save_run_in(n_steps, T0, T1, dir):

    # Create run.in file for GPUMD
    run_in = f"""
        potential ../../nep.txt

        velocity {T0}
        time_step 1

        compute_extrapolation asi_file ../../active_set.asi check_interval 10 gamma_low 2 gamma_high 10
        dump_thermo 200
        ensemble npt_mttk tri 0 0 temp {T0} {T1}
        run {n_steps}

        compute_extrapolation asi_file ../../active_set.asi check_interval 10 gamma_low 2 gamma_high 10
        dump_thermo 200
        ensemble npt_mttk tri 0 0 temp {T1} {T0}
        run {n_steps}
    """

    with open(os.path.join(dir, "run.in"), "w") as f:
        text = "\n".join(line.strip() for line in run_in.splitlines())
        f.write(text)
