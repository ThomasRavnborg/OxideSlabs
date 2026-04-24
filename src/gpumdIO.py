
import os

def save_run_in(parameters_md, dir):

    try:
        temperature = parameters_md['temperature']
    except KeyError:
        temperature = 300
    try:
        n_steps = parameters_md['n_steps']
    except KeyError:
        n_steps = 10000
    try:
        dump_interval = parameters_md['dump_interval']
    except KeyError:
        dump_interval = 100

    # Create run.in file for GPUMD
    run_in = f"""
        potential ../../nep.txt

        velocity {temperature}
        time_step 1.0
        dump_exyz {dump_interval} 0 1

        ensemble npt_mttk iso 0 0 temp {temperature} {temperature+200}
        run {n_steps}

        ensemble npt_mttk iso 0 0 temp {temperature+200} {temperature}
        run {n_steps}
    """
    with open(os.path.join(dir, "run.in"), "w") as f:
        text = "\n".join(line.strip() for line in run_in.splitlines())
        f.write(text)