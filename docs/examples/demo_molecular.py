"""
======================================================================
Space filling model actor
======================================================================

A small example to show how to use the space_filling_model actor and
and generate a space filling model for a protein. This example also
shows how to parse PDBx/mmCIF files to obtain atomic info essential
(coordinates and element names) to constructing the model.

Importing necessary modules
"""

import urllib
import os
from fury import window, actor, ui, molecular

###############################################################################
# Downloading the PDBx file whose model is to be rendered.
# User can change the pdbx_code depending on which protein they want to
# visualize
pdbx_code = '4hhb'
downloadurl = "https://files.rcsb.org/download/"
pdbxfn = pdbx_code + ".cif"
flag = 0
if not os.path.isfile(pdbxfn):
    flag = 1
    url = downloadurl + pdbxfn
    outfnm = os.path.join(pdbxfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
    except Exception:
        print("Error in downloading the file!")


molecule = molecular.Molecule()
table = molecular.PeriodicTable()

###############################################################################
# Parsing the mmCIF file for information about coordinates and atoms

pdbxfile = open(pdbxfn, 'r')
pdbx_lines = pdbxfile.readlines()
for line in pdbx_lines:
    _line = line.split()
    try:
        if _line[0] == 'ATOM' or _line[0] == 'HETATM':

            # obtain coordinates of atom
            coorX, coorY, coorZ = float(_line[10]), float(_line[11]), \
                                  float(_line[12])

            # obtain the atomic number of atom
            atomic_num = table.get_atomic_number(_line[2])

            # add the atomic data to the molecule
            molecular.add_atom(molecule, atomic_num, coorX, coorY, coorZ)

    except Exception:
        continue


###############################################################################
# Doing 5 things here -
# 1. Creating a scene object
# 2. configuring the camera's position
# 3. Creating and adding axes actor to the scene
# 4. Computing the bonding information for the molecule.
# 4. Generating and adding molecular model to the scene.

scene = window.Scene()
scene.set_camera(position=(20, 20, 0), focal_point=(0, 0, 0),
                 view_up=(0, 1, 0))
axes_actor = actor.axes()
scene.add(axes_actor)
molecular.compute_bonding(molecule)


# stick representation
scene.add(molecular.molecular_stick_rep_actor(molecule, bond_thickness=2))

# ball and stick representation
# scene.add(molecular.molecular_bstick_rep_actor(molecule,
#                                                atom_scale_factor=0.3,
#                                                bond_thickness=2))

# sphere representation
# scene.add(molecular.molecular_sphere_rep_actor(molecule))

###############################################################################
# Dimensions of the output screen
screen_x_dim = 600
screen_y_dim = 600
dims = (screen_x_dim, screen_y_dim)

###############################################################################
# creating a ShowManager object
showm = window.ShowManager(scene, size=dims, reset_camera=True,
                           order_transparent=True)

tb = ui.TextBlock2D(text=pdbx_code, position=(screen_x_dim/2-40,
                    screen_y_dim/12), font_size=30, color=(1, 1, 1))
scene.add(tb)

###############################################################################
# Delete the PDBx file if it's downloaded from the internet
if flag:
    os.remove(outfnm)

interactive = False
if interactive:
    window.show(scene, size=dims, title=pdbx_code)
window.record(scene, size=dims, out_path=pdbx_code+'.png')
