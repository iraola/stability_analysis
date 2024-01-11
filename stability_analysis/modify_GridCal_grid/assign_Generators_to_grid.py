import numpy as np
import pandas as pd
import numpy as np
import GridCalEngine.api as gce


def assign_StaticGen(GridCal_grid, d_raw_data, d_op):
    
    for bus in GridCal_grid.get_buses():
        if bus.is_slack:
            bus_code=int(bus.code)
            gen=bus.generators[0]
            
            gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
            gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
            gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
            gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
            gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
            gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
            gen.Vset=1
            
        elif len(bus.generators)!=0:
            
            bus_code=int(bus.code)
            gen_name=bus.generators[0].name
                        
            GridCal_grid.add_static_generator(bus, gce.StaticGenerator(gen_name, P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG'], Q=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'QG']))
            
            bus.generators=[]
            
    
def assign_PV_or_StaticGen(GridCal_grid,d_raw_data,d_op):
    for bus in GridCal_grid.get_buses():
        bus_code=int(bus.code)
        
        if bus_code in list(d_raw_data['generator']['I']):
            if bus.is_slack:
                gen=bus.generators[0]

                gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
                gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
                gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
                gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
                gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
                gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
                gen.Vset=1
                
            else:
                bus.generators=[]
                bus.static_generators=[]
                if d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'Static']==1:
                    gen_name=str(bus.code)#generators[0].name
                                
                    GridCal_grid.add_static_generator(bus, gce.StaticGenerator(gen_name, P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG'], Q=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'QG']))#,power_factor=1)
                                        
                elif d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'Static']==0:
                    gen_name=str(bus.code)#generators[0].name
                    
                    gen = gce.Generator(gen_name)#voltage_module=1.0,p_min=-Sn_vsc,p_max=Sn_vsc,Snom=Sn_vsc)
                    
                    gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
                    gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
                    gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
                    gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
                    gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
                    gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
                    gen.Vset=1
                    
                    GridCal_grid.add_generator(bus, gen)
                    
def assign_PV_or_StaticGen_read_original_grid(GridCal_grid,d_raw_data,d_op):
    for bus in GridCal_grid.get_buses():
        bus_code=int(bus.code)
        
        if bus_code in list(d_raw_data['generator']['I']):
            if bus.is_slack:
                gen=bus.generators[0]
                
                gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
                gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
                gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
                gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
                gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
                gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
                gen.Vset=1
                
                
                
            elif d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'Static']==1:
                gen_name=bus.generators[0].name
                            
                GridCal_grid.add_static_generator(bus, gce.StaticGenerator(gen_name, P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG'], Q=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'QG']))#,power_factor=1)
                
                bus.generators=[]
                
            elif d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'Static']==0:
                gen=bus.generators[0]
                
                gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
                gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
                gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
                gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
                gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
                gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
                gen.Vset=1 
                
def assign_PVGen(GridCal_grid,d_raw_data,d_op):
    for bus in GridCal_grid.get_buses():
        bus_code=int(bus.code)
        
        if bus_code in list(d_raw_data['generator']['I']):
            if bus.is_slack:
                gen=bus.generators[0]

                gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
                gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
                gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
                gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
                gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
                gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
                gen.Vset=1
                
            else:
                bus.generators=[]
                bus.static_generators=[]
                                          
                gen_name=str(bus.code)#generators[0].name
                
                gen = gce.Generator(gen_name)#voltage_module=1.0,p_min=-Sn_vsc,p_max=Sn_vsc,Snom=Sn_vsc)
                
                gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
                gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
                gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
                gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
                gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
                gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
                gen.Vset=1
                
                GridCal_grid.add_generator(bus, gen)
                