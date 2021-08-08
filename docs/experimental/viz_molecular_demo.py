"""
======================================================================
Space filling model actor
======================================================================

A small example to show how to use the space_filling_model actor and
and generate a space filling model for a protein. This example also
shows how to parse PDBx/mmCIF files to obtain atomic info essential
(coordinates and element names) to construct the model.

Importing necessary modules
"""

import urllib
import os
from fury import window, actor, ui, molecular as mol
import numpy as np

###############################################################################
# Downloading the PDB file whose model is to be rendered.
# User can change the pdb_code depending on which protein they want to
# visualize
pdb_code = '4ury'
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
# creating a PeriodicTable() object to obtain atomic numbers from names of
# elements
table = mol.PeriodicTable()

###############################################################################
# Creating empty lists which will be filled with atomic information as we
# parse the pdb file.
NumberOfAtoms = 0

points = []
elements = []
atom_names = []
model = []
sheets = []
helix = []
residue_seq = []
chain = []
is_hetatm = []
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
                coorX, coorY, coorZ = float(line[6]), float(line[7]), \
                                      float(line[8])
                resi = line[5]
                current_chain = ord(line[4])
                points += [[coorX, coorY, coorZ]]
                residue_seq += [resi]
                chain += [current_chain]
                elements += [table.atomic_number(line[-1])]
                atom_names += [line[2]]
                model += [current_model_number]
                NumberOfAtoms += 1
                if(line[0] == 'HETATM'):
                    is_hetatm += [1]
                else:
                    is_hetatm += [0]
        if line[0] == 'SHEET':
            start_chain = ord(line[5])
            start_resi = int(line[6])
            end_chain = ord(line[8])
            end_resi = int(line[9])
            r = [start_chain, start_resi, end_chain, end_resi]
            sheets += [r]
        if line[0] == 'HELIX':
            start_chain = ord(line[4])
            start_resi = int(line[5])
            end_chain = ord(line[7])
            end_resi = int(line[8])
            r = [start_chain, start_resi, end_chain, end_resi]
            helix += [r]
    except Exception:
        continue

points = np.array(points)
residue_seq = np.array(residue_seq, dtype=int)
chain = np.array(chain)
elements = np.array(elements)
atom_names = np.array(atom_names)
model = np.array(model)
sheets = np.array(sheets)
helix = np.array(helix)
is_hetatm = np.array(is_hetatm)


###############################################################################
# Doing 5 things here -
# 1. Creating a scene object
# 2. Configuring the camera's position
# 3. Creating and adding axes actor to the scene
# 4. Computing the bonding information for the molecule.
# 5. Generating and adding molecular model to the scene.

scene = window.Scene()
scene.set_camera(position=(20, 10, 0), focal_point=(0, 0, 0),
                 view_up=(0, 1, 0))
scene.zoom(0.8)
axes_actor = actor.axes()
scene.add(axes_actor)
molecule = mol.Molecule(elements, points, atom_names, model,
                        residue_seq, chain, sheets, helix, is_hetatm)
mol.compute_bonding(molecule)

# stick representation
scene.add(mol.stick(molecule, bond_thickness=0.2))

# ribbon representation
scene.add(mol.ribbon(molecule), mol.bounding_box(molecule))

# ball and stick representation
# scene.add(mol.ball_stick(molecule, atom_scale_factor=0.3,
#                          bond_thickness=0.2))

# sphere representation
# scene.add(mol.sphere_cpk(molecule))

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
