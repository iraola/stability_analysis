def assign_PQ_load(GridCal_grid,d_raw_data):#, solved_point=None, d_pf=None):
    for load in GridCal_grid.get_loads():
        bus_name=int(load.bus.code)
        #if solved_point==None:
        load.P=d_raw_data['load'].loc[d_raw_data['load'].query('I == @bus_name').index[0],'PL']
        load.Q=d_raw_data['load'].loc[d_raw_data['load'].query('I == @bus_name').index[0],'QL']
        # else:
        #     load.P=d_pf['pf_load'].loc[d_pf['pf_load'].query('bus == @bus_name').index[0],'P']*100
        #     load.Q=d_pf['pf_load'].loc[d_pf['pf_load'].query('bus == @bus_name').index[0],'Q']*100
    
        load.B=0
        load.Ir=0