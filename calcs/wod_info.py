import numpy as np

wodvars = ['1-Temperature','2-Salinity','3-Oxygen','4-Phosphate',
           '5-Total-Phosphorus','6-Silicate','7-Nitrite',
           '8-Nitrate','9-pH','10-Ammonia','11-Chlorophyll',
           '12-Phaeophytin','13-PrimaryProd','14-Biochem',
           '15-LightC14','16-DarkC14','17-Alkalinity',
           '18-POC','19-DOC','20-pCO2','21-TCO2','22-XCO2sea',
           '23-NO2NO3','24-Transmissivity','25-Pressure',
           '26-Conductivity','33-Tritium','34-Helium',
           '35-DeltaHe','36-DeltaC14','37-DeltaC13',
           '38-Argon','39-Neon','40-CFC11',
           '41-CFC12','42-CFC113','43-O18']

wodvars_attrs = {
    'time'        : {'units' : 'days since 0001-01-01 00:00:00',
                     'calendar' : 'gregorian'},
    'depth'       : {'long_name':'Depth','units':'m'},
    'lat'         : {'long_name':'Latitude','units':'degrees_north'},
    'lon'         : {'long_name':'Longitude','units':'degrees_east'},
    'temperature' : {'units' : 'degree C',
                     'long_name' : 'Temperature'},
    'salinity'    : {'units' : '',
                     'long_name' : 'Salinity'},
    'oxygen'      : {'units' : 'ml/l',
                     'long_name' : 'Dissolved oxygen'},
    'phosphate'    : {'units' : '$\mu$M',
                      'long_name' : 'Phosphate'},
    'nitrate'    : {'units' : '$\mu$M',
                    'long_name' : 'Nitrate'},
    'silicate'    : {'units' : '$\mu$M',
                     'long_name' : 'Silicate'}}

mlperl_2_mmolm3 = 1.e6 / 1.e3 / 22.3916

wodvars_convert_units = {'oxygen':{'factor':mlperl_2_mmolm3,
                                   'new_units':'mmol/m^3'}}

woa_shortnames = {'temperature':'t',
                  'salinity':'s',
                  'nitrate':'n',
                  'oxygen':'o',
                  'o2sat':'O',
                  'AOU':'A',
                  'silicate':'i',
                  'phosphate':'p'}


def compute_depth_bounds(standard_z):
    x = np.vstack((standard_z[:-1],standard_z[1:])).mean(axis=0)
    b = np.vstack( (np.concatenate((standard_z[0:1],x)), np.concatenate((x,standard_z[-1:])))).T
    return b

standard_z = np.concatenate((
    np.arange(0.,105.,5.),
    np.arange(125.,525.,25.),
    np.arange(550.,2050.,50.),
    np.arange(2100.,5600.,100.)))

standard_z_depth_bounds = compute_depth_bounds(standard_z)
