from GridCalEngine.IO.file_handler import FileOpen
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions
from GridCalEngine.Simulations.PowerFlow.power_flow_options import ReactivePowerControlMode, SolverType
from GridCalEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver

import numpy as np
import pandas as pd
from os import path


def create_model(path_raw, name_raw=None):
    
    # GET GRID TOPOLOGY
    if name_raw is not None:
        raw_file = path.join(path_raw, name_raw)
    else:
        raw_file = path_raw
    grid = FileOpen(raw_file).open()
    return grid
    
def run_powerflow(grid,solver_type=SolverType.NR, Qconrol_mode=ReactivePowerControlMode.Direct):
            
    # RUN POWERFLOW
    for solver_type in [SolverType.IWAMOTO, SolverType.NR, SolverType.LM, SolverType.FASTDECOUPLED]:
    
        print(solver_type)
    
        options = PowerFlowOptions(solver_type,
                                   verbose=False,
                                   initialize_with_existing_solution=False,
                                   retry_with_other_methods=False,
                                   ignore_single_node_islands = True,
                                   control_q=Qconrol_mode,
                                   tolerance = 1e-6)
                                   # max_iter=20)
        
        pf = PowerFlowDriver(grid, options)
        pf.run()
        
        print('Converged:', pf.convergence_reports[0].converged_[0])
    
    # grid.get_buses()
    return pf

def preprocess_grid(gridCal_grid, d_raw_data):

    df_lines_grid = pd.DataFrame(columns=['bf','bt'])
    df_trafos_grid = pd.DataFrame(columns=['bf','bt'])
    
    
    if len(gridCal_grid.get_branches()) > len(d_raw_data['branch'])+len(d_raw_data['trafo']):
        for idx, line in enumerate(gridCal_grid.lines):
            df_lines_grid.loc[idx,'bf']=line.bus_from.code
            df_lines_grid.loc[idx,'bt']=line.bus_to.code
    # Ensure you're comparing with consistent data types
    df_lines_grid[['bf', 'bt']] = df_lines_grid[['bf', 'bt']].astype(int)
    df_lines_draw = d_raw_data['branch'][['I', 'J']].astype(int)
    
    # Create sets of tuples for fast comparison
    lines_set_grid = set(map(tuple, df_lines_grid[['bf', 'bt']].values))
    lines_set_draw = set(map(tuple, df_lines_draw[['I', 'J']].values))
    
    # Compute the difference
    missing_lines = lines_set_grid - lines_set_draw
    
    # Convert back to DataFrame for inspection
    missing_df = pd.DataFrame(list(missing_lines), columns=['bf', 'bt'])
    
    print(missing_df)
