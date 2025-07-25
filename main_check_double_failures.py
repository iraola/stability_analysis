#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import GridCalEngine.api as gce
from pprint import pprint
import json
import networkx as nx
import numpy as np

import os
import pandas as pd
from stability_analysis.preprocess import preprocess_data, read_data, \
    process_raw
from stability_analysis.powerflow import GridCal_powerflow, process_powerflow, slack_bus, fill_d_grid_after_powerflow
from GridCalEngine.Simulations.PowerFlow.power_flow_options import ReactivePowerControlMode, SolverType
from stability_analysis.preprocess import parameters

from stability_analysis.state_space import build_ss, generate_NET, generate_elements
from stability_analysis.analysis import small_signal
from small_signal_analysis import *
from stability_analysis.modify_GridCal_grid import assign_Generators_to_grid, assign_PQ_Loads_to_grid, assign_SlackBus_to_grid

def detect_islands(grid):
    """
    Builds a graph with the buses as nodes and the active branches as edges,
    and returns True if there is more than one connected component
    (i.e. at least one island), False otherwise.
    """
    G = nx.Graph()
    # Add buses
    for bus in grid.buses:
        G.add_node(bus)
    # Add edges only for active lines
    for line in grid.lines:
        if line.active:
            i = line.bus_from
            j = line.bus_to
            G.add_edge(i, j)
    # Count components
    components = list(nx.connected_components(G))
    return len(components) > 1


def check_line_overloads(results, grid):
    """
    Checks every line in the grid for overload conditions.
    Uses the 'loading' array from the results object and the branch data
    provided by grid.get_branches().

    Returns:
      - A list of line indices where loading > 1 (i.e., overloaded lines).
    """
    overloaded_lines = []
    for idx, (loading, branch) in enumerate(zip(results.loading, grid.get_branches())):
        if loading > 1:
            overloaded_lines.append(idx)
    return overloaded_lines


def check(grid):
    """
    Verifies that no component remains deactivated at the end of all simulations.
    Raises an Exception if any line, transformer, or generator is still inactive.
    """
    for idx, line in enumerate(grid.lines):
        if not line.active:
            raise Exception(f"Line at index {idx} is not active")
    for idx, transformer in enumerate(grid.transformers2w):
        if not transformer.active:
            raise Exception(f"Transformer at index {idx} is not active")
    for idx, generator in enumerate(grid.generators):
        if not generator.active:
            raise Exception(f"Generator at index {idx} is not active")
            
def read_excel_sheets_as_dict(file_path):
    """
    Reads an Excel file with multiple sheets and returns a dictionary.
    
    Parameters:
        file_path (str): Path to the Excel file.
    
    Returns:
        dict: A dictionary where keys are sheet names and values are DataFrames.
    """
    xls = pd.read_excel(file_path, sheet_name=None)
    return xls    

#%%

if __name__ == "__main__":
    
#%%    
    # Path to the grid file
    #GRID_FILE = './stability_analysis/data/raw/IEEE118_NREL_stable_grid.gridcal'#'IEEE118_opf.gridcal'
    GRID_TOPOLOGY = './stability_analysis/data/raw/IEEE118busNREL.raw'#'IEEE118_opf.gridcal'
    '''
     Number of lines: 175
     Number of generators: 53
     Number of transformers: 11
    '''
    # Alternative example:
    # GRID_FILE = 'grids/IEEE_14.xlsx'

    # Open the grid
    #grid = gce.open_file(GRID_FILE)
    #grid_topology = GridCal_powerflow.create_model(GRID_TOPOLOGY)
    
    # for line in grid_topology.lines:
    #    grid.add_line(line)
    # for trafo in grid_topology.transformers2w:
    #     grid.add_transformer2w(trafo)
 
    
    grid= GridCal_powerflow.create_model(GRID_TOPOLOGY)

    #grid.lines=grid_topology.lines
    #grid.transformers2w =  grid_topology.transformers2w
    
#%%    
    # ----------------------------------------------------------------
    filename= './stability_analysis/data/cases/IEEE118_NREL_stable_'
    d_grid = read_excel_sheets_as_dict(filename+'d_grid.xlsx')
    d_raw_data = read_excel_sheets_as_dict(filename+'d_raw_data.xlsx')
    d_opf = read_excel_sheets_as_dict(filename+'d_opf.xlsx')
    d_op = read_excel_sheets_as_dict(filename+'d_op.xlsx')
    
    
    assign_Generators_to_grid.assign_PVGen(GridCal_grid=grid, d_raw_data=d_raw_data, d_op=d_op,
                                            voltage_profile_list=True, solved_point=True, d_pf=d_opf)

    assign_PQ_Loads_to_grid.assign_PQ_load(grid, d_raw_data)#, solved_point=True, d_pf=d_opf)

    # for bus in grid.buses:
    #     bus_num=int(bus.code)
    #     idx=d_opf['pf_bus'].query('bus == @bus_num').index[0]

    #     bus.Vm0=d_opf['pf_bus'].loc[idx,'Vm']
    #     bus.Va0=d_opf['pf_bus'].loc[idx,'theta']/180*np.pi
        
    # slack_bus_num = d_grid['T_global'].loc[0,'ref_bus']
    # assign_SlackBus_to_grid.assign_slack_bus(grid, slack_bus_num)

 #%%   
    # # -----------------------------------------------------------------
    #excel_headers = "IEEE_118_FULL_headers"
    excel_data = "IEEE_118_FULL"
    # excel_op = "OperationData_IEEE_118_NREL"
    excel_lines_ratings = "IEEE_118_Lines"

    path_data='./stability_analysis/data/'
    #excel_sys = os.path.join(path_data, "cases", excel_headers + ".xlsx")
    excel_sg = os.path.join(path_data, "cases", excel_data + "_data_sg.xlsx")
    excel_vsc = os.path.join(path_data, "cases", excel_data + "_data_vsc.xlsx")
    # excel_op = os.path.join(path_data, "cases", excel_op + ".xlsx")
    
    excel_lines_ratings = os.path.join(
            path_data, "cases", excel_lines_ratings + ".csv")

    
    #d_raw_data = process_raw.read_raw(GRID_FILE)
    # d_op = read_data.read_data(excel_op)

    
    # # FOR the 118-bus system
    # d_raw_data['generator']['Region'] = d_op['Generators']['Region']
    # d_raw_data['load']['Region'] = d_op['Loads']['Region']
    # # d_raw_data['branch']['Region']=1
    # d_raw_data['results_bus']['Region'] = d_op['Buses']['Region']
    # d_raw_data['generator']['MBASE'] = d_op['Generators']['Snom']
    lines_ratings = pd.read_csv(excel_lines_ratings)
    
    for line in grid.lines:
        bf = int(line.bus_from.code)
        bt = int(line.bus_to.code)
        line.rate = lines_ratings.loc[
            lines_ratings.query('Bus_from == @bf and Bus_to == @bt').index[
                0], 'Max Flow (MW)']

    for trafo in grid.transformers2w:
        bf = int(trafo.bus_from.code)
        bt = int(trafo.bus_to.code)
        trafo.rate = lines_ratings.loc[
            lines_ratings.query('Bus_from == @bf and Bus_to == @bt').index[
                0], 'Max Flow (MW)']

    # Preprocess input raw data to match Excel file format
    # preprocess_data.preprocess_raw(d_raw_data)
    
    # d_grid, d_grid_0 = read_data.read_sys_data(excel_sys)
    d_sg = read_data.read_data(excel_sg)
    d_vsc = read_data.read_data(excel_vsc)

    
    # idx_sg0=list(d_op['Generators'].query('Snom_SG==0')['BusNum'])
    # d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @idx_sg0').index,'alpha_P_SG']=0
    
    # idx_cig0=list(d_op['Generators'].query('Snom_CIG==0')['BusNum'])
    # d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @idx_cig0').index,'alpha_P_GFOR']=0
    # d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @idx_cig0').index,'alpha_P_GFOL']=0
    
    # for i in d_raw_data['generator'].index:
    #     alphas= d_raw_data['generator'].loc[i, [col for col in d_raw_data['generator'].columns if col.startswith('alpha') ]]
    #     nan_indices = alphas[alphas.isna()].index
    #     d_raw_data['generator'].loc[i,nan_indices]=1/(alphas.isna().sum())
        
    # for el in ['GFOL','GFOR']:
    #     d_op['Generators']['Snom_'+el]=d_raw_data['generator']['alpha_P_'+el]*d_op['Generators']['Snom_CIG']
        
#%%
    # ----------------------------------------------------------------
    # Print the total counts of each component type
    num_lines = len(grid.lines)
    num_generators = len(grid.generators)
    num_transformers = len(grid.transformers2w)
    print(f"Number of lines: {num_lines}")
    print(f"Number of generators: {num_generators}")
    print(f"Number of transformers: {num_transformers}")

#%%
    # ----------------------------------------------------------------

    # Probability of failure (100% means any component you deactivate will fail)
    FAILURE_PROBABILITY = 100

    # Prepare results buckets for first- and second-level failures
    results = {
        'first_level_line': [],
        'first_level_transformer': [],
        'first_level_generator': [],
        'second_level_line_line': [],
        'second_level_line_transformer': [],
        'second_level_line_generator': [],
        'second_level_transformer_line': [],
        'second_level_transformer_transformer': [],
        'second_level_transformer_generator': [],
        'second_level_generator_line': [],
        'second_level_generator_transformer': [],
        'second_level_generator_generator': [],
    }

#%%
    # --- Simulate base case ---
    # Calculate Power Flow
    pf_results = GridCal_powerflow.run_powerflow(grid,Qconrol_mode=ReactivePowerControlMode.NoControl)
    
    # P=list(d_raw_data['load']['PL'])
    # Q=list(d_raw_data['load']['QL'])
    
    # for i in range(81):#Olen(P)):
    #     grid.loads[i].P=float(P[i])
    #     grid.loads[i].Q=float(Q[i])
        
    #     # if i >=80:
    
    # nc = compile_numerical_circuit_at(grid)
    # nc.generator_data.cost_0[:] = 0
    # nc.generator_data.cost_1[:] = 0
    # nc.generator_data.cost_2[:] = 0
    # pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8, control_q=ReactivePowerControlMode.Direct)#, max_iter=100)
    # opf_options = gce.OptimalPowerFlowOptions(solver=gce.SolverType.NR, verbose=0, ips_tolerance=1e-4, ips_iterations=50)

    # pf_results = multi_island_pf_nc(nc=nc, options=pf_options)

    # d_opf_results = ac_optimal_power_flow(nc= nc,
    #                                       pf_options= pf_options,
    #                                       opf_options= opf_options,
    #                                       # debug: bool = False,
    #                                       #use_autodiff = True,
    #                                       pf_init= True,
    #                                       Sbus_pf= pf_results.Sbus,
    #                                       voltage_pf= pf_results.voltage,
    #                                       plot_error= True)

    
    
    if pf_results.convergence_reports[0].converged_[0]:
        print('Base case power flow converges')
        # Update PF results and operation point of generator elements
        d_pf = process_powerflow.update_OP(grid, pf_results, d_raw_data)
        #d_opf = process_optimal_power_flow.update_OP(grid, d_opf_results, d_raw_data)
        
        stability, T_EIG = calculate_small_signal(d_raw_data,d_op, grid, d_grid, d_sg, d_vsc, d_opf)
        
    # else:
    #     print('Base case power flow does not converge')
        
#%%        
    # --- Simulate first-level failures on lines ---
     #for idx, line in enumerate(grid.lines):

        #line.active = False
        grid.lines[0].active = False
        grid.lines[11].active = False
        
        pf_results = GridCal_powerflow.run_powerflow(grid,Qconrol_mode=ReactivePowerControlMode.NoControl)

        if pf_results.convergence_reports[0].converged_[0]:
            print('Base case power flow converges')
            # Update PF results and operation point of generator elements
            d_pf = process_powerflow.update_OP(grid, pf_results, d_raw_data)
            #d_opf = process_optimal_power_flow.update_OP(grid, d_opf_results, d_raw_data)
            
            stability, T_EIG = calculate_small_signal(d_raw_data,d_op, grid, d_grid, d_sg, d_vsc, d_opf)
       
            
            # Otherwise, simulate second-level failures:

            # # 1) Second-level failures on other lines
            # for idx2, line2 in enumerate(grid.lines):
            #     if idx2 != idx:
            #         line2.active = False
            #         if not gce.power_flow(grid).converged:
            #             results['second_level_line_line'].append([idx, idx2, detect_islands(grid)])
            #         line2.active = True

            # # 2) Second-level failures on transformers
            # for idx2, transformer in enumerate(grid.transformers2w):
            #     transformer.active = False
            #     if not gce.power_flow(grid).converged:
            #         results['second_level_line_transformer'].append([idx, idx2, detect_islands(grid)])
            #     transformer.active = True

            # # 3) Second-level failures on generators
            # for idx2, generator in enumerate(grid.generators):
            #     generator.active = False
            #     if not gce.power_flow(grid).converged:
            #         results['second_level_line_generator'].append([idx, idx2, detect_islands(grid)])
            #     generator.active = True
            

        line.active = True

    # Ensure all components are active again
    check(grid)
    print('Lines done')
    # --- Simulate first-level failures on transformers ---
    for idx, transformer in enumerate(grid.transformers2w):
        transformer.active = False
        if not gce.power_flow(grid).converged:
            results['first_level_transformer'].append([idx, detect_islands(grid)])
        else:
            # Second-level on lines
            for idx2, line in enumerate(grid.lines):
                line.active = False
                if not gce.power_flow(grid).converged:
                    results['second_level_transformer_line'].append([idx, idx2, detect_islands(grid)])
                line.active = True

            # Second-level on other transformers
            for idx2, transformer2 in enumerate(grid.transformers2w):
                if idx2 != idx:
                    transformer2.active = False
                    if not gce.power_flow(grid).converged:
                        results['second_level_transformer_transformer'].append([idx, idx2, detect_islands(grid)])
                    transformer2.active = True

            # Second-level on generators
            for idx2, generator in enumerate(grid.generators):
                generator.active = False
                if not gce.power_flow(grid).converged:
                    results['second_level_transformer_generator'].append([idx, idx2, detect_islands(grid)])
                generator.active = True

        transformer.active = True

    check(grid)
    print('Transformers done')
    # --- Simulate first-level failures on generators ---
    for idx, generator in enumerate(grid.generators):
        generator.active = False
        if not gce.power_flow(grid).converged:
            results['first_level_generator'].append([idx, detect_islands(grid)])
        else:
            # Second-level on lines
            for idx2, line in enumerate(grid.lines):
                line.active = False
                if not gce.power_flow(grid).converged:
                    results['second_level_generator_line'].append([idx, idx2, detect_islands(grid)])
                line.active = True

            # Second-level on transformers
            for idx2, transformer in enumerate(grid.transformers2w):
                transformer.active = False
                if not gce.power_flow(grid).converged:
                    results['second_level_generator_transformer'].append([idx, idx2, detect_islands(grid)])
                transformer.active = True

            # Second-level on other generators
            for idx2, generator2 in enumerate(grid.generators):
                if idx2 != idx:
                    generator2.active = False
                    if not gce.power_flow(grid).converged:
                        results['second_level_generator_generator'].append([idx, idx2, detect_islands(grid)])
                    generator2.active = True

        generator.active = True

    check(grid)
    print('Generators done')
    # Print the aggregated results
    pprint(results)
    with open('results.json', 'w', encoding='utf-8') as f:
        # 2) Dump the `results` dict as pretty-printed JSON
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Results saved to results.json")
