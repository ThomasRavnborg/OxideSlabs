

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
    N_rep = 20

    #if ensemble == 'nve':
        # Reduce number of replications
        #N_rep = 5

    if bulk:
        directions = ['tri']
    else:
        directions = ['x', 'y', 'xy']

    direction_p1_p2 = ' '.join([f'{dir} 0 0' for dir in directions])

    if T1 != T0:
        run_in = ""
        n_steps = n_steps // 2
    else:
        #run_in = ""
        
        if bulk:
            run_in = f"""
                replicate {N_rep} {N_rep} {N_rep}
                dump_xyz -1 1 1 supercell.xyz
                run 0
                """
        else:
            run_in = f"""
                replicate {N_rep} {N_rep} 1
                dump_xyz -1 1 1 supercell.xyz
                run 0
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

            print(f'Maximum frequency that can be resolved is {500.0 / (dt * delta_dump)} THz.')
            run_in += f"""
                dump_thermo {n_steps//10000}
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

