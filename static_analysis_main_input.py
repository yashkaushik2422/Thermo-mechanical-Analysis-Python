import numpy as np
import matplotlib.pyplot as plt
import PyFEMP
import Q1_LE_Dyn_T_Commented as ELEMENT

#Temperature valujes
REFT = 20 + 273.5      # reference temperature
outerT = 56 + 273.5    # outside temperature
innerT = 20  + 273.5   # inner temperature
DTout = outerT - REFT  # Delta tem in the outside area
DTin = innerT - REFT   # Delta tem in the inside area


# Dealing with materials in PyFEMP
FEM = PyFEMP.FEM_Simulation(ELEMENT)
a, n = 0.3, 15                          # Insulation layer thickness and element count per direction
x_lenth = a + 0.8                       # the total length for the x direction
y_lenth = a + 0.3                       # the total length for the y direction
x_n = int(n * 2)                        # the n factor to divide the x direction by (must be integer)
y_n = int(x_n * y_lenth / x_lenth )     # the n factor to divide the y direction by (must be integer)

# generate mesh
XI, Elem = PyFEMP.msh_rec([0.0, 0.0], [0.8 + a, 0.3 + a], [x_n, y_n], type='Q1')   # making the mesh element
FEM.Add_Mesh(XI, Elem)   # adding the mesh element
# set list of material parameter for each material

#           E      , nu  , rho , c_m , bx , by ,  c_a , c_T  , r  , T0     , c_Tu  , cs
# Elem  [  "E"  , "nu","rho","c_m","bx","by", "c_a", "c_T", "r",  "T0"  , "c_Tu", "cs"]
steel = [213.0e9, 0.25, 7900, 0   , 0  , 0  , 48   ,   0  , 0  , REFT   , 22e-6 , 0   ]

# Elem [ "E"  , "nu","rho","c_m","bx","by", "c_a", "c_T", "r", "T0"   , "c_Tu", "cs"]
foam = [38.0e6, 0.24, 52  , 900 , 0  , 0  , 20e-3, 0    , 0  , REFT   , 4.2e-5, 0   ]


# Set material for each element
xEL = x_lenth/x_n    # mesh element length in x direction
yEL = y_lenth/y_n    # mesh element length in y direction

# adding the material for the elements
for i, elmt in enumerate(Elem):                                     # looping throw all mesh elements
    # c o m p u t e c e n t e r o f e l e m e n t
    elmtcentr = np.mean(XI[elmt], axis=0)                           # getting the element center coordination
    if elmtcentr[0] >= a+(xEL/2) and elmtcentr[1] <= 0.3-(yEL/2):   # if the element is in the steel area
        FEM.Add_Material(steel)                                     # adding the steel material
    else:                                                           # else if it's not in the steel material
        FEM.Add_Material(foam)                                      # add the foam material


# Boundary conditions

#fixing the steel wall below
FEM.Add_EBC(f"y<=0e-10 and x >= {a} and x <= {a} + 0.2 ", "UY", 0)
FEM.Add_EBC(f"y<=0e-10 and x >= {a} and x <= {a} + 0.2", "UX", 0)
#fixing right edge in x-direction
FEM.Add_EBC(f"x== {a} +0.8 and y<= 0.3 ", "UX", 0)


# FEM.Add_NBC() (No NBCs apply)

# adding the temperature change from the resting reference temperature
FEM.Add_EBC("x == 0", "DT", DTout)
FEM.Add_EBC(f"y == {a} + 0.3", "DT", DTout)
FEM.Add_EBC(f"x >= {a}+0.2 and y==0", "DT", DTin)

# Perform analysis
FEM.Analysis()
FEM.NextStep(1.0, 1.0)

#Print residual
print(FEM.NewtonIteration())
print(FEM.NewtonIteration())
print(FEM.NewtonIteration())


# Get stress value in the steel element
location_list = []  # to keep the locations of the stress values if needed
sigall_list = []  # to store the stress values in a
for i, elmt in enumerate(Elem):  # looping throw all steel element
    # compute center of element
    elmtcentr = np.mean(XI[elmt], axis=0)
    if elmtcentr[0] >= a+(xEL/2) and elmtcentr[1] <= 0.3-(yEL/2):   # if the element is steel
        Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0], elmtcentr[1]])  # in the center
        location_list.append(Location)   # adding the location of the stress to the matrix
        sigall_list.append(sig_all)      # adding the stress to the matrix
        Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0]-(x_lenth/(x_n*2)), elmtcentr[1]-(y_lenth/(y_n*2))])  # in the lower left corner
        location_list.append(Location)   # adding the location of the stress to the matrix
        sigall_list.append(sig_all)      # adding the stress to the matrix
        Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0]+(x_lenth/(x_n*2)), elmtcentr[1]-(y_lenth/(y_n*2))])  # in the lower right corner
        location_list.append(Location)   # adding the location of the stress to the matrix
        sigall_list.append(sig_all)      # adding the stress to the matrix
        Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0]-(x_lenth/(x_n*2)), elmtcentr[1]+(y_lenth/(y_n*2))])  # in the upper left corner
        location_list.append(Location)   # adding the location of the stress to the matrix
        sigall_list.append(sig_all)      # adding the stress to the matrix
        Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0]+(x_lenth/(x_n*2)), elmtcentr[1]+(y_lenth/(y_n*2))])  # in the upper right corner
        location_list.append(Location)   # adding the location of the stress to the matrix
        sigall_list.append(sig_all)      # adding the stress to the matrix

#Extract maximum stress and its corresponding location
max_stress_index = np.argmax(sigall_list)
max_stress_location = location_list[max_stress_index]
max_stress_value = max(sigall_list) * 1e-3  # Convert stress to MPa
print("Max stress is {:.4f} KPa at location: {}".format(max_stress_value, max_stress_location))


#Plot Results
fig, ax = plt.subplots(1, 2, figsize=(21.0, 7.5))
postplot1 = FEM.ShowMesh(ax[0], deformedmesh=True, PostName="T",boundaryconditions=True)
ax[0].set_xlim(0, a + 0.8)
ax[0].set_ylim(0, a + 0.3)
cbar1 = fig.colorbar(postplot1, ax=ax[0])
cbar1.set_label('absolute temperature $ {\\theta}$ in K')
postplot2 = FEM.ShowMesh(ax[1], deformedmesh=True, PostName="SigMises")
ax[0].set_xlim(0, a + 0.8)
ax[0].set_ylim(0, a + 0.3)
cbar2 = fig.colorbar(postplot2, ax=ax[1])
cbar2.set_label('von Mises stress $   igma_{VM}$ in Pa')
plt.figtext(0.7, 0.9, f'Max stress: {max_stress_value:.2f} KPa\nLocation: {max_stress_location}', fontsize=12, ha='right', va='top', color='red')
plt.show()



