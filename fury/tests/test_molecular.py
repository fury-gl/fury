import numpy.testing as npt
import numpy as np
from fury import window, actor, molecular

def test_periodic_table():
    # testing class PeriodicTable()
    table = molecular.PeriodicTable()
    npt.assert_equal(table.atomic_number('C'), 6)
    npt.assert_equal(table.element_name(7), 'Nitrogen')
    npt.assert_equal(table.atomic_symbol(8), 'O')
    npt.assert_almost_equal(table.atomic_radius(1, 'VDW'), 1.2)
    npt.assert_almost_equal(table.atomic_radius(6, 'Covalent'), 0.75)
    npt.assert_array_almost_equal(table.atom_color(1), np.array([1, 1, 1]))

def get_default_molecular_info():
    elements = np.array([6, 6, 1, 1, 1, 1, 1, 1])
    atom_coords = np.array([[0.5723949486E+01, 0.5974463617E+01, 0.5898320525E+01],
                            [0.6840181327E+01, 0.6678078649E+01, 0.5159998484E+01],
                            [0.4774278044E+01, 0.6499436628E+01, 0.5782310182E+01],
                            [0.5576295333E+01, 0.4957554302E+01, 0.5530844713E+01],
                            [0.5926818174E+01, 0.5907771848E+01, 0.6968386044E+01],
                            [0.6985130929E+01, 0.7695511362E+01, 0.5526416671E+01],
                            [0.7788135127E+01, 0.6150201159E+01, 0.5277430519E+01],
                            [0.6632858893E+01, 0.6740709254E+01, 0.4090898288E+01]]
                           )
    return elements, atom_coords

def test_molecule_creation():
    elements, atom_coords = get_default_molecular_info()
    molecule = molecular.Molecule(elements=elements, coords=atom_coords)
    npt.assert_array_almost_equal(molecular.get_atomic_number_array(molecule),
                                  elements)
    npt.assert_array_almost_equal(molecular.get_atomic_position_array(molecule), atom_coords)

def test_add_atom_bond_creation():
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 6, 0, 0, 0)
    molecular.add_atom(molecule, 6, 1, 0, 0)
    molecular.add_bond(molecule, 0, 1, 1)
    npt.assert_equal(molecular.get_total_num_bonds(molecule), 1)
    npt.assert_equal(molecular.get_total_num_atoms(molecule), 2)

def test_atomic_number():
    # testing atomic number get/set functions
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 4, 0, 0, 0)

    # testing get_atomic_number
    npt.assert_equal(molecular.get_atomic_number(molecule, 0), 4)

    # testing set_atomic_number
    molecular.set_atomic_number(molecule, 0, 6)
    npt.assert_equal(molecular.get_atomic_number(molecule, 0), 6)


def test_atomic_position():
    # testing atomic position get/set functions
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 4, 0, 0, 0)

    # testing get_atomic_position
    npt.assert_array_almost_equal(molecular.get_atomic_position(molecule, 0),
                                                                np.array([0, 0, 0]))

    # testing set_atomic_number
    molecular.set_atomic_position(molecule, 0, 1, 1, 1)
    npt.assert_array_almost_equal(molecular.get_atomic_position(molecule, 0),
                                                                np.array([1, 1, 1]))

def test_bond_type():
    # testing bond type get/set functions

    # testing get_bond_type
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 6, 0, 0, 0)
    molecular.add_atom(molecule, 6, 1, 0, 0)
    molecular.add_bond(molecule, 0, 1, 3)
    npt.assert_equal(molecular.get_bond_type(molecule, 0), 3)

    # testing set_bond_type
    molecular.set_bond_type(molecule, 0, 2)
    npt.assert_equal(molecular.get_bond_type(molecule, 0), 2)

    # testing get_bond_types_array
    npt.assert_array_almost_equal(molecular.get_bond_types_array(molecule),
                                                                 np.array([2]))

def test_deep_copy():
    molecule1 = molecular.Molecule()
    molecular.add_atom(molecule1, 6, 0, 0, 0)
    molecular.add_atom(molecule1, 6, 1, 0, 0)
    molecular.add_bond(molecule1, 0, 1, 1)
    molecule2 = molecular.Molecule()
    molecular.deep_copy(molecule2, molecule1)
    npt.assert_equal(molecular.get_total_num_bonds(molecule2), 1)
    npt.assert_equal(molecular.get_total_num_atoms(molecule2), 2)

def test_compute_bonding():
    elements, atom_coords = get_default_molecular_info()
    molecule = molecular.Molecule(elements=elements, coords=atom_coords)
    molecular.compute_bonding(molecule)
    npt.assert_equal(molecular.get_total_num_bonds(molecule), 7)


def test_make_molecular_viz_aesthetic():
    centers = np.zeros((1, 3))
    box = actor.box(centers=centers)
    molecular.make_molecularviz_aesthetic(box)
    npt.assert_equal(box.GetProperty().GetDiffuse(), 1)
    npt.assert_equal(box.GetProperty().GetSpecular(), 1)
    npt.assert_equal(box.GetProperty().GetAmbient(), 0.3)
    npt.assert_equal(box.GetProperty().GetSpecularPower(), 100.0)


def test_sphere_rep_actor(interactive=False):
    elements, atom_coords = get_default_molecular_info()
    molecule = molecular.Molecule(elements=elements, coords=atom_coords)
    table = molecular.PeriodicTable()
    colormodes = ['discrete', 'single']
    colors = np.array([[table.atom_color(1), table.atom_color(6)],
                       [[150/255, 250/255, 150/255]]], dtype=object)
    for i, colormode in enumerate(colormodes):
        test_actor = molecular.sphere_rep_actor(molecule, colormode)

        scene = window.Scene()
        scene.add(test_actor)
        scene.reset_camera()
        scene.reset_clipping_range()

        if interactive:
            window.show(scene)

        npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr,
                                         colors=colors[i])
        npt.assert_equal(report.objects, 1)


# def test_bstick_rep_actor(interactive=False):
#     elements, atom_coords = get_default_molecular_info()
#     molecule = molecular.Molecule(elements=elements, coords=atom_coords)
#     molecular.compute_bonding(molecule)
#     colormodes = ['discrete', 'single']
#     atom_scale_factor = [0.3, 0.4]
#     bond_thickness = [1, 1.2]
#     multiple_bonds = ['On', 'Off']
#     table = molecular.PeriodicTable()
#     colors = np.array([[table.atom_color(1), table.atom_color(6)],
#                        [[150/255, 250/255, 150/255],
#                         [50/255, 50/255, 50/255]]], dtype=object)
#     for i, colormode in enumerate(colormodes):
#         test_actor = molecular.bstick_rep_actor(molecule, colormode,
#                                                 atom_scale_factor[i],
#                                                 bond_thickness[i],
#                                                 multiple_bonds[i])
#         scene = window.Scene()
#         scene.add(test_actor)
#         scene.reset_camera()
#         scene.reset_clipping_range()

#         if interactive:
#             window.show(scene)

#         npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

#         arr = window.snapshot(scene)
#         report = window.analyze_snapshot(arr,
#                                          colors=colors[i])
#         npt.assert_equal(report.objects, 1)

