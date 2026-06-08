

def create_run_in(ensemble='npt_ber', dt=1, n_steps=5*1e5, n_dump=1000, T0=300, T1=300, bulk=True):
    
    # Replace underscores with spaces in ensemble name for pimd ensembles
    if ensemble.split('_')[0] == 'pimd':
        ensemble = ensemble.replace('_', ' ')

    # Ensure n_steps and n_dump are integers
    n_steps = int(n_steps)
    n_dump = int(n_dump)
    delta_dump = n_steps // n_dump

    if T1 != T0:
        n_steps = n_steps // 2

    # Thermostat and barostat parameters
    T_coup = 100
    p_coup = 1000

    # Target pressures in GPa
    p_xx = p_yy = p_zz = 0
    p_yz = p_xz = p_xy = 0
    # Elastic constants in GPa
    C_xx = C_yy = C_zz = 282
    C_yz = C_xz = C_xy = 104

    if not bulk:
        # Set z-components (direction of vacuum) of the elastic constants above 2000 GPa for slabs
        # By doing this, the coupling constant for that component will be zero and that box component will not be changed 
        C_zz = C_yz = C_xz = 2001

    # Pressure control parameters
    pressure_control_params = f"{p_xx} {p_yy} {p_zz} {p_yz} {p_xz} {p_xy} {C_xx} {C_yy} {C_zz} {C_yz} {C_xz} {C_xy} {p_coup}"

    if bulk:
        directions = ['tri']
    else:
        directions = ['x', 'y', 'xy']

    direction_p1_p2 = ' '.join([f'{dir} 0 0' for dir in directions])

    """
    if T1 != T0:
        run_in = "potential ../../../nep.txt"
        n_steps = n_steps // 2
    else:
        
        if bulk:
            run_in = f"#replicate {N_rep} {N_rep} {N_rep}"
        else:
            run_in = f"#replicate {N_rep} {N_rep} 1"
        run_in += f"potential ../../../nep.txt"
    """
    
    def _setup_single_run(run_in, ensemble, T0, T1):

        if ensemble == 'nve':

            print(f'Maximum frequency that can be resolved is {500.0 / (dt * delta_dump)} THz.')
            run_in += f"""
                dump_thermo {delta_dump//10}
                ensemble nvt_lan {T0} {T1} {T_coup}
                run {n_steps//10}
            """
            # production
            run_in += f"""
                dump_thermo {delta_dump}
                dump_netcdf {delta_dump}
                ensemble nve
                run {n_steps}
            """
            return run_in

        run_in += f"""
            dump_thermo {delta_dump//10}
            dump_netcdf {delta_dump}"""

        if ensemble.split('_')[0] == 'pimd':
            run_in += f"""
            dump_beads {delta_dump}"""

        if ensemble.split('_')[0] == 'nvt':
            run_in += f"""
            ensemble {ensemble} {T0} {T1} {T_coup}"""

        elif ensemble == 'npt_mttk':
            run_in += f"""
            ensemble {ensemble} temp {T0} {T1} tperiod {T_coup} {direction_p1_p2} pperiod {p_coup}"""

        elif ensemble.split('_')[0] == 'npt' or ensemble.split(' ')[0] == 'pimd':
            run_in += f"""
            ensemble {ensemble} {T0} {T1} {T_coup} {pressure_control_params}"""
        
        run_in += f"""
            run {n_steps}
        """
        return run_in
    
    # Define potential line for run.in with the path to the NEP model file
    run_in = "potential ../../../nep.txt"

    # Initialize velocity and time step
    run_in += f"""
        velocity {T0}
        time_step {dt}
        """
    
    # Setup first run from T0 to T1
    run_in = _setup_single_run(run_in, ensemble, T0, T1)
    # Setup second run from T1 to T0 if T1 != T0
    if T1 != T0:
        run_in = _setup_single_run(run_in, ensemble, T1, T0)
    
    # Strip leading/trailing whitespace and extra indentation from run_in
    run_in_text = "\n".join(line.strip() for line in run_in.splitlines())
    return run_in_text

