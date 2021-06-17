import numpy as np
from fury import window, actor

def ball_and_stick(atom_coords, elem_names):
    atomic_radius = {'H':0.38, 'C':0.77}
    error_factor = 0.05
    atomic_radius = {k:v + error_factor for k,v in atomic_radius.items()}

    atom_index = np.arange(0, len(elem_names))
    atoms = elem_names

    atoms_rad = [atomic_radius[x] for x in atoms]
    # print(atoms_rad, atom_index)

    i_atom = np.copy(atom_index)
    p = np.copy(atom_coords)
    p_compare = p
    r = atoms_rad
    r_compare = r

    source_row = np.arange(len(atoms))
    max_atoms = len(atoms)

    bonds = np.zeros((len(atoms)+1, max_atoms+1), dtype=np.int8)
    bond_dists = np.zeros((len(atoms)+1, max_atoms+1), dtype=np.float32)

    # Algorithm to generate bonding
    for i in range(max_atoms-1):
        p_compare = np.roll(p_compare, -1, axis=0)
        #m_compare = np.roll(m_compare, -1, axis=0)
        r_compare = np.roll(r_compare, -1, axis=0)
        mask = 0
        dists = np.linalg.norm(p - p_compare, axis=1) #* mask
        r_bond = r + r_compare
        bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

        source_row = source_row
        target_row = source_row + i + 1 # Will be out of bounds of bonds array for some values of i
        target_row = np.where(np.logical_or(target_row > len(atoms), mask==1), len(atoms), target_row) #If invalid target, write to dummy row

        source_atom = i_atom
        target_atom = i_atom + i + 1 # Will be out of bounds of bonds array for some values of i
        target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==1), max_atoms, target_atom) #If invalid target, write to dummy col

        bonds[(source_row, target_atom)] = bond
        bonds[(target_row, source_atom)] = bond
        bond_dists[(source_row, target_atom)] = dists
        bond_dists[(target_row, source_atom)] = dists

    bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row
    bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col
    bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row
    bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col

    # print('Counting and condensing bonds')

    bonds_numeric = [[i for i,x in enumerate(row) if x] for row in (bonds)]
    bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate((bond_dists))]
    n_bonds = [len(x) for x in bonds_numeric]
    # print(n_bonds)
    bond_data = {'bonds':bonds_numeric, 'n_bonds':n_bonds, 'bond_lengths':bond_lengths}
    # print(bond_data['bonds'])

    # offsets for double or triple bonds
    def offsets(atom_coords, c1_coord, c2_coord, bond_type):
        ## initial assumption that the molecule is planar in nature
        planar = 1
        ## finding equation of plane on which the atoms reside
        p1, p2, p3 = atom_coords[:3]
        v1 = p3 - p1
        v2 = p2 - p1
        eq = np.cross(v1, v2)
        d = np.dot(eq, p3)
        if not np.any(eq):
            planar = 0 # if the vector of plane has coefficients (0, 0, 0) then the atoms do not lie on same plane

        c1 = c1_coord
        c2 = c2_coord
        dir_vector = c2_coord - c1_coord
        offset = bond_type * 0.04
        if planar:
            sol = np.cross(dir_vector, eq)
            sol = sol / np.linalg.norm(sol)     # coverting into unit vector
            if bond_type==2 or bond_type==3:
                c1_l = c1 - sol*offset
                c1_u = c1 + sol*offset
                c2_l = c2 - sol*offset
                c2_u = c2 + sol*offset
                return c1_l, c1_u, c2_l, c2_u


    # generate bonds
    # print(len(bond_data['bonds']))
    bond_coords = []
    bond_colors = []
    cpkr = {'H': [1, 1, 1, 1.2], 'C': [144/255, 144/255, 144/255, 1.7]}
    indexes_done = []
    i = 0
    c = 0
    #print(len(bond_data['bonds']))
    for index, (bonds,ename) in enumerate(zip(bond_data['bonds'], elem_names)):
        for bond in bonds:
            if bond not in indexes_done:
                print(index, ename, elem_names[bond], i)#, atom_coords[bond], atom_coords[index])
                if(elem_names[bond]!= ename):
                    bond_colors.append(cpkr[ename][:3])
                    bond_colors.append(cpkr[elem_names[bond]][:3])
                    mid = (atom_coords[index] + atom_coords[bond])/2
                    bond_coords.append([atom_coords[index], mid])
                    bond_coords.append([mid, atom_coords[bond]])
                else:
                    if bond_data['bond_lengths'][index][i] < 1.22:
                        bond_type = 3
                    elif bond_data['bond_lengths'][index][i] < 1.36:
                        bond_type = 2
                    else:
                        bond_type = 1
                    if bond_type == 1 or bond_type == 3:
                        bond_coords.append([atom_coords[bond], atom_coords[index]])
                        bond_colors.append(cpkr[ename][:3])
                    if bond_type == 2 or bond_type == 3:
                        c1_l, c1_u, c2_l, c2_u = offsets(atom_coords, atom_coords[index], atom_coords[bond], bond_type)
                        bond_colors.append(cpkr[ename][:3])
                        bond_colors.append(cpkr[ename][:3])
                        bond_coords.append([c1_l, c2_l])
                        bond_coords.append([c1_u, c2_u])

            i += 1
        indexes_done += [index]

    unique_elem_types = np.unique(elem_names)
    atom_colors = np.ones((len(atom_coords), 3))
    radii = np.ones((len(atom_coords), 3))
    for i, typ in enumerate(unique_elem_types):
        atom_colors[elem_names == typ] = cpkr[typ][:3]
        radii[elem_names == typ] = cpkr[typ][-1]/4

    sticks = actor.streamtube(bond_coords, colors=bond_colors, linewidth=0.05)
    balls = actor.sphere(atom_coords, colors=atom_colors, radii=radii, phi=32, theta=32)
    return sticks, balls


scene = window.Scene()
#scene.add(ball, sticks)
scene.background((1, 1, 1))
scene.set_camera(position=(0, 0, 100), focal_point=(0, 0, 0),
                 view_up=(0, 1, 0))

# ethane (single bonds)
atom_coords = np.array([[0.5723949486E+01, 0.5974463617E+01, 0.5898320525E+01],
                        [0.6840181327E+01, 0.6678078649E+01, 0.5159998484E+01],
                        [0.4774278044E+01, 0.6499436628E+01, 0.5782310182E+01],
                        [0.5576295333E+01, 0.4957554302E+01, 0.5530844713E+01],
                        [0.5926818174E+01, 0.5907771848E+01, 0.6968386044E+01],
                        [0.6985130929E+01, 0.7695511362E+01, 0.5526416671E+01],
                        [0.7788135127E+01, 0.6150201159E+01, 0.5277430519E+01],
                        [0.6632858893E+01, 0.6740709254E+01, 0.4090898288E+01]], dtype=float)
elem_names = np.array(['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'])

sticks, balls = ball_and_stick(atom_coords, elem_names)
scene.add(sticks, balls)

# ethene (double bonds)
atom_coords = np.array([[0.5449769880E+01, 0.5680940296E+01, 0.5519555369E+01],
                        [0.6346918574E+01, 0.6272796762E+01, 0.6280955432E+01],
                        [0.4603843481E+01, 0.6218164558E+01, 0.5109232482E+01],
                        [0.5522586159E+01, 0.4630394155E+01, 0.5265590478E+01],
                        [0.6275632424E+01, 0.7323831630E+01, 0.6534741329E+01],
                        [0.7193811558E+01, 0.5736436394E+01, 0.6691459388E+01]], dtype=float)
elem_names = np.array(['C', 'C', 'H', 'H', 'H', 'H'])

sticks, balls = ball_and_stick(atom_coords+[0, 0, 5], elem_names)
scene.add(sticks, balls)


# ethene (triple bonds)
atom_coords = np.array([[0.5899518696E+01, 0.5868718390E+01, 0.5737443048E+01],
                        [0.6573681090E+01, 0.6576391430E+01, 0.6424519094E+01],
                        [0.5300877605E+01, 0.5237561906E+01, 0.5127884913E+01],
                        [0.7173348068E+01, 0.7204870233E+01, 0.7035206023E+01]], dtype=float)
elem_names = np.array(['C', 'C', 'H', 'H'])
bond_type = np.array([3, 1, 3, 1, 1, 1]) # ethyne
sticks, balls = ball_and_stick(atom_coords+[0, 0, 10], elem_names)
scene.add(sticks, balls)


window.show(scene, size=(500, 500))
