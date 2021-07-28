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
import numpy as np

###############################################################################
# Downloading the PDB file whose model is to be rendered.
# User can change the pdb_code depending on which protein they want to
# visualize
pdb_code = '1pgb'
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
# creating an PeriodicTable() object to obtain atomic numbers from names of
# elements
table = molecular.PeriodicTable()

###############################################################################
# Creating empty lists which will be filled with atomic information as we
# parse the pdb file.
NumberOfAtoms = 0

Points = []
AtomType = []
AtomTypeStrings = []
Model = []
Sheets = []
Helix = []
Residue = []
Chain = []
IsHetatm = []
SecondaryStructures = []
current_model_number = 1

###############################################################################
# Parsing the pdb file for information about coordinates and atoms

pdbfile = open(pdbfn, 'r')
pdb_lines = pdbfile.readlines()
for line in pdb_lines:
    line = line.split()
    try:
        if line[0] == 'ATOM' or line[0] == 'HETATM':
            if line[-1] != 'H':
                # print(2)
                coorX, coorY, coorZ = float(line[6]), float(line[7]), \
                                      float(line[8])
                resi = line[5]
                chain = ord(line[4])
                Points += [[coorX, coorY, coorZ]]
                Residue += [resi]
                Chain += [chain]
                AtomType += [table.atomic_number(line[-1])]
                AtomTypeStrings += [line[2]]
                Model += [current_model_number]
                NumberOfAtoms += 1
                if(line[0] == 'HETATM'):
                    IsHetatm += [1]
                else:
                    IsHetatm += [0]
        if line[0] == 'SHEET':
            startChain = ord(line[5])
            startResi = int(line[6])
            endChain = ord(line[8])
            endResi = int(line[9])
            r = [startChain, startResi, endChain, endResi]
            Sheets += [r]
        if line[0] == 'HELIX':
            startChain = ord(line[4])
            startResi = int(line[5])
            endChain = ord(line[7])
            endResi = int(line[8])
            r = [startChain, startResi, endChain, endResi]
            Helix += [r]
    except Exception:
        continue


Points = np.array(Points)
Residue = np.array(Residue, dtype=int)
Chain = np.array(Chain)
AtomType = np.array(AtomType)
AtomTypeStrings = np.array(AtomTypeStrings)
Model = np.array(Model)
Sheets = np.array(Sheets)
Helix = np.array(Helix)
IsHetatm = np.array(IsHetatm)


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
molecule = molecular.Molecule(AtomType, Points, AtomTypeStrings, Model,
                              Residue, Chain, Sheets, Helix, IsHetatm)
molecular.compute_bonding(molecule)


# stick representation
# scene.add(molecular.stick_rep_actor(molecule, bond_thickness=2))

# ribbon representation
scene.add(molecular.ribbon_rep_actor(molecule))

# ball and stick representation
# scene.add(molecular.bstick_rep_actor(molecule, atom_scale_factor=0.3,
#                                      bond_thickness=2))

# sphere representation
# scene.add(molecular.sphere_rep_actor(molecule))

###############################################################################
# Dimensions of the output screen
screen_x_dim = 600
screen_y_dim = 600
dims = (screen_x_dim, screen_y_dim)

###############################################################################
# creating a ShowManager object
showm = window.ShowManager(scene, size=dims, reset_camera=True,
                           order_transparent=True)

tb = ui.TextBlock2D(text=pdb_code, position=(screen_x_dim/2-40,
                    screen_y_dim/12), font_size=30, color=(1, 1, 1))
scene.add(tb)

###############################################################################
# Delete the PDBx file if it's downloaded from the internet
if flag:
    os.remove(outfnm)

interactive = True
if interactive:
    window.show(scene, size=dims, title=pdb_code)
window.record(scene, size=dims, out_path=pdb_code+'.png')
