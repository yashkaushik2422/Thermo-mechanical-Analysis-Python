import numpy as np
import matplotlib.pyplot as plt
import PyFEMP
import Q1_LE_Dyn_T_Commented as ELEMENT

REFT = 20   # reference temperature
outerT = 96   # outside temperature
innerT = 20   # inner temperature
DTout = outerT - REFT  # Delta tem in the outside area
DTin = innerT - REFT   # Delta tem in the inside area890-




# Listing 1: Dealing with materials in PyFEMP
FEM = PyFEMP.FEM_Simulation(ELEMENT, verbose=False)
a, n = 0.2, 5
x_lenth = a + 0.8
y_lenth = a + 0.3
x_n = int(n * 3)
y_n = int(n)
FEM.verbose_system = False

# g e n e r a t e m e s h
XI, Elem = PyFEMP.msh_rec([0.0, 0.0], [0.8 + a, 0.3 + a], [x_n, y_n])
FEM.Add_Mesh(XI, Elem, verbose=False)
# s e t l i s t o f m a t e r i a l p a r a m e t e r f o r e a c h m a t e r i a l
# Elem m [ "E"     , "nu","rho","c_m","bx","by", "c_a", "c_T", "r",    "T0", "c_Tu", "cs"]
steel = [213.0e9   , 0.25, 7900, 0   , 0  , 0  , 48   , 500  , 0  , REFT   , 22e-6 , 0]

# Elem m[ "E"    , "nu","rho","c_m","bx","by", "c_a", "c_T", "r", "T0"   , "c_Tu", "cs"]
foam = [38.0e6   , 0.24, 52  , 800 , 0  , 0  , 20e-3, 2300 , 0  , REFT   , 4.2e-5, 0]


# s e t m a t e r i a l f o r e a c h e l e m e n t s u c c e s s i v e l y
xEL = x_lenth/x_n  # mesh element length in x direction
yEL = y_lenth/y_n    # mesh element length in y direction

for i, elmt in enumerate(Elem):
    # c o m p u t e c e n t e r o f e l e m e n t
    elmtcentr = np.mean(XI[elmt], axis=0)
    if elmtcentr[0] >= a+(xEL/2) and elmtcentr[1] <= 0.3-(yEL/2):
        FEM.Add_Material(steel)
    else:
        FEM.Add_Material(foam)

# adding the boundary conditions  !!!!!! need more work
FEM.Add_EBC(f"y==0 and x >= {a} and x <= {a} + 0.2 ", "UY", 0)
FEM.Add_EBC(f"y==0 and x >= {a} and x <= {a} + 0.2", "UX", 0)
# fixing the right wall    !!! check if we need to do this or not ***********
FEM.Add_EBC(f"x== {a} +0.8 and y<= 0.3 ", "UX", 0)


#   add a NBC later pls
# FEM.Add_NBC()

# adding the temperature change from the resting reference temperature
FEM.Add_EBC("x == 0", "DT", DTout)
FEM.Add_EBC(f"y == {a} + 0.3", "DT", DTout)
FEM.Add_EBC(f"x >= {a}+0.2 and y==0", "DT", DTin)  # check if we add x< a+0.8 will change anything

FEM.Analysis()

time = 0
dt = 30*60
N_steps = 50 # change this later as needed





# record array
uy_vs_t = np.zeros((N_steps+1,2))

for step in range(N_steps):
    time += dt
    FEM.NextStep(time, 1.0)
    FEM.NewtonIteration()
    FEM.NewtonIteration()


    # getting the stress value for each mesh area in the center and the corners
    location_list = []  # to keep the locations of the stress values if needed
    sigall_list = []  # to store the stress values in a
    for i, elmt in enumerate(Elem):
        # c o m p u t e c e n t e r o f e l e m e n t
        elmtcentr = np.mean(XI[elmt], axis=0)
        if elmtcentr[0] >= a + (xEL / 2) and elmtcentr[1] <= 0.3 - (yEL / 2):
            Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0], elmtcentr[1]])  # in the center

            location_list.append(Location)
            sigall_list.append(sig_all)
            Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0] - (x_lenth / (x_n * 2)), elmtcentr[1] - (
                        y_lenth / (y_n * 2))])  # in the lower left corner
            location_list.append(Location)
            sigall_list.append(sig_all)
            Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0] + (x_lenth / (x_n * 2)), elmtcentr[1] - (
                        y_lenth / (y_n * 2))])  # in the lower right corner
            location_list.append(Location)
            sigall_list.append(sig_all)
            Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0] - (x_lenth / (x_n * 2)), elmtcentr[1] + (
                        y_lenth / (y_n * 2))])  # in the upper left corner
            location_list.append(Location)
            sigall_list.append(sig_all)
            Location, sig_all = FEM.PostProcessing("SigMises", [elmtcentr[0] + (x_lenth / (x_n * 2)), elmtcentr[1] + (
                        y_lenth / (y_n * 2))])  # in the upper right corner
            location_list.append(Location)
            sigall_list.append(sig_all)

    max_stress_index = np.argmax(sigall_list)
    max_stress_location = location_list[max_stress_index]
    max_stress_value = max(sigall_list) * 1e-3  # Convert stress to kPa
    print("Max stress is {:.4f} kPa at location: {}".format(max_stress_value, max_stress_location))
    print(f"at time = {time/60/60} h")
    uy_vs_t[step + 1] = np.array([time / 60/60, max_stress_value])

    if step == 0:
        fig, ax = plt.subplots(1, 3, figsize=(18.0, 6.0))

    ax[0].cla()
    postplot1 = FEM.ShowMesh(ax[0], deformedmesh=True, PostName="T")
    ax[0].set_xlim(0, a + 0.8)
    ax[0].set_ylim(0, a + 0.3)
    cbar = fig.colorbar(postplot1,ax=ax[0])
    cbar.set_label('absolute temperature $ {\\theta}$ in K')
    ax[0].set_aspect(9 / 9.5)

    ax[1].cla()
    postplot2 = FEM.ShowMesh(ax[1], deformedmesh=True, PostName="SigMises")
    ax[1].set_xlim(0, a + 0.8)
    ax[1].set_ylim(0, a + 0.3)
    cbar1 = fig.colorbar(postplot2, ax=ax[1])
    cbar1.set_label('von Mises stress $sigma_{VM}$ in Pa')
    ax[1].set_aspect(9 / 9.5)

    ax[2].cla()
    ax[2].plot(uy_vs_t[:step + 1, 0], uy_vs_t[:step + 1, 1])
    ax[2].set_xlabel('t- time in H')
    ax[2].set_ylabel('von Mises stress $sigma_{VM}$ in kPa')
    ax[2].set_xlim(0, N_steps * dt/60/60)
    ax[2].set_ylim(0, 500)
    #ax[2].set_aspect(9 / 9.5)

    fig.tight_layout()
    plt.pause(0.001)

    if step< N_steps-1:
        cbar.remove()
        cbar1.remove()


plt.show()






'''
    if step == 0:
        fig, ax = plt.subplots(1, 2, figsize=(21.0, 7.5))


    ax[0].cla()
    ax[1].cla()
    postplot1 = FEM.ShowMesh(ax[0], deformedmesh=True, PostName="T")
    ax[0].set_xlim(0, a + 0.8)
    ax[0].set_ylim(0, a + 0.3)
    cbar1 = fig.colorbar(postplot1, ax=ax[0])
    cbar1.set_label('absolute temperature $ {\\theta}$ in K')



    postplot2 = FEM.ShowMesh(ax[1], deformedmesh=True, PostName="SigMises")
    ax[0].set_xlim(0, a + 0.8)
    ax[0].set_ylim(0, a + 0.3)
    cbar2 = fig.colorbar(postplot2, ax=ax[1])
    cbar2.set_label('von Mises stress $   igma_{VM}$ in MPa')
    fig.tight_layout()
    plt.pause(0.00001)
    cbar1.remove()
    cbar2.remove()


plt.show()




'''


















'''
# this part will show what element are assigned to                                   
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# Create a new figure for the material plot
fig, ax = plt.subplots(figsize=(10.5, 7.5))

# Create an array to store the material of each element
material = []

# Iterate over the elements and append the material to the array
for i, elmt in enumerate(Elem):
    elmtcentr = np.mean(XI[elmt], axis=0)
    if elmtcentr[0] > a and elmtcentr[1] < 0.3:
        material.append(0)  # steel
    else:
        material.append(1)  # foam

# Convert the material array to a numpy array
material = np.array(material)

# Create a color map
cmap = mcolors.ListedColormap(['red', 'blue'])  # red for steel, blue for foam

# Plot the mesh with the material color
for i in range(len(Elem)):
    xi = XI[Elem[i]]
    ax.add_patch(patches.Polygon(xi, facecolor=cmap(material[i])))

# Set the colorbar with the correct labels
cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1])
cbar.set_label('Material')
cbar.set_ticklabels(['steel', 'foam'])

# Show the plot
plt.show()

'''
