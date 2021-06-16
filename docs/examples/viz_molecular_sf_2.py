"""
======================================================================
Space filling model actor
======================================================================

A small example to show how to use the space_filling_model actor and
and generate a space filling model for a protein. This example also
shows how to parse PDB files via Biopython to obtain atomic info
essential (coordinates and element names) to constructing the model.

Importing necessary modules
"""

import numpy as np
import urllib
import os
from fury import window, actor, ui
from Bio.PDB import PDBParser

###############################################################################
# Downloading the PDB file whose model is to be rendered.
# User can change the pdb_code depending on which protein they want to
# visualize
pdb_code = '1fdh'
downloadurl = "https://files.rcsb.org/download/"
pdbfn = pdb_code + ".pdb"
flag = 0
if not os.path.isfile(pdbfn):
    flag = 1
    url = downloadurl + pdbfn
    outfnm = os.path.join(pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
    except Exception:
        print("Error in downloading the file!")

###############################################################################
# list to store names of the elements
elem_sym_list = []
# list to store coordinates of the atoms
atom_coords = []

# parsing the PDB file for information about coordinates and atoms
pdb_parser = PDBParser(PERMISSIVE=1)
structure_pdb = pdb_parser.get_structure(pdb_code, pdbfn)
pdbxfile = open(pdbfn, 'r')
for model in structure_pdb:
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom_coords.append(atom.coord)
                elem_sym_list.append(atom.element)

elem_sym_list = np.array(elem_sym_list)
atom_coords = np.array(atom_coords)

###############################################################################
# can un-comment the line below to check the number of atoms being rendered
# print(len(atom_coords))

###############################################################################
# Doing 4 things here -
# 1. Creating a scene object
# 2. configuring the camera's position
# 3. Creating and adding axes actor to the scene
# 4. Generating and adding the space-filling model to the scene
scene = window.Scene()
scene.set_camera(position=(20, 20, 0), focal_point=(0, 0, 0),
                 view_up=(0, 1, 0))
axes_actor = actor.axes()
scene.add(axes_actor)
sf_model, elements = actor.molecular_sf(atom_coords, elem_sym_list,
                                        return_unique_elements=True)
scene.add(sf_model)

###############################################################################
# Dimensions of the output screen
screen_x_dim = 600
screen_y_dim = 600
# ini variable will help in creating the key for elements and their colors
ini = 30

###############################################################################
# Arranging the elements in reverse alphabetic order
elements = np.array(elements, dtype=object)
elements = elements[elements[:, 0].argsort()[::-1]]


###############################################################################
# creating a ShowManager object
showm = window.ShowManager(scene,
                           size=(screen_x_dim, screen_y_dim),
                           reset_camera=True, order_transparent=True)

###############################################################################
# Key of elements (with the respective colors)
for i, element in enumerate(elements):
    tb = ui.TextBlock2D(text='  ' + element[0],
                        position=(screen_x_dim-60, ini), font_size=20,
                        color=(1, 1, 1))
    scene.add(tb)
    tb = ui.TextBlock2D(text='  ', position=(screen_x_dim-70, ini+3),
                        font_size=20, bg_color=element[1], color=(0, 0, 0))
    scene.add(tb)
    ini += 30
tb = ui.TextBlock2D(text='Elements', position=(screen_x_dim-100, ini),
                    font_size=20, color=(1, 1, 1))
scene.add(tb)
tb = ui.TextBlock2D(text=pdb_code,
                    position=(screen_x_dim/2-40, screen_y_dim/12),
                    font_size=30, color=(1, 1, 1))
scene.add(tb)

###############################################################################
# Delete the PDB file if it's downloaded from the internet
if flag:
    os.remove(outfnm)

interactive = False
if interactive:
    window.show(scene, title=pdb_code, size=(screen_x_dim, screen_y_dim))
window.record(showm.scene, size=(600, 600), out_path=pdb_code+'.png')
