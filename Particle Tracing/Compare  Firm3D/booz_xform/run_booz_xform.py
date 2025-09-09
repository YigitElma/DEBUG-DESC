import booz_xform as bx

wout_filename = 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'
boozmn_filename = 'boozmn_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'
mboz = 96
nboz = 96

b = bx.Booz_xform()
b.read_wout(wout_filename,True)
b.mboz = mboz
b.nboz = nboz
b.run()
b.write_boozmn(boozmn_filename)
