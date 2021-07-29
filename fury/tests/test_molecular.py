import numpy.testing as npt
import numpy as np
from fury import window, actor, molecular


def test_periodic_table():
    # Testing class PeriodicTable()
    table = molecular.PeriodicTable()
    npt.assert_equal(table.atomic_number('C'), 6)
    npt.assert_equal(table.element_name(7), 'Nitrogen')
    npt.assert_equal(table.atomic_symbol(8), 'O')
    npt.assert_allclose(table.atomic_radius(1, 'VDW'), 1.2, 0.1, 0)
    npt.assert_allclose(table.atomic_radius(6, 'Covalent'), 0.75, 0.1, 0)
    npt.assert_array_almost_equal(table.atom_color(1), np.array([1, 1, 1]))

    # Test errors
    npt.assert_raises(ValueError, table.atomic_radius, 4, "test")


def get_default_molecular_info(all_info=False):
    elements = np.array([6, 6, 1, 1, 1, 1, 1, 1])
    atom_coords = np.array([[0.5723949486E+01, 0.5974463617E+01,
                             0.5898320525E+01],
                            [0.6840181327E+01, 0.6678078649E+01,
                             0.5159998484E+01],
                            [0.4774278044E+01, 0.6499436628E+01,
                             0.5782310182E+01],
                            [0.5576295333E+01, 0.4957554302E+01,
                             0.5530844713E+01],
                            [0.5926818174E+01, 0.5907771848E+01,
                             0.6968386044E+01],
                            [0.6985130929E+01, 0.7695511362E+01,
                             0.5526416671E+01],
                            [0.7788135127E+01, 0.6150201159E+01,
                             0.5277430519E+01],
                            [0.6632858893E+01, 0.6740709254E+01,
                             0.4090898288E+01]]
                           )
    atom_types = np.array(['CA', 'CA', 'H', 'H', 'H', 'H', 'H', 'H'])
    model = np.ones(8)
    residue = np.ones(8)
    chain = np.ones(8)*65
    is_hetatm = np.ones(8, dtype=bool)
    sheet = []
    helix = []
    if all_info:
        return elements, atom_coords, atom_types, model, residue, chain, \
            is_hetatm, sheet, helix
    return elements, atom_coords


def test_molecule_creation():
    elements, atom_coords = get_default_molecular_info()
    molecule = molecular.Molecule(elements=elements, coords=atom_coords)
    npt.assert_array_almost_equal(molecular.get_atomic_number_array(molecule),
                                  elements)
    npt.assert_array_almost_equal(molecular.get_atomic_position_array
                                  (molecule), atom_coords)

    # Test errors
    elements = np.array([6, 6])
    npt.assert_raises(ValueError, molecular.Molecule, elements, atom_coords)

    elements = [i for i in range(8)]
    npt.assert_raises(ValueError, molecular.Molecule, elements, atom_coords)


def test_add_atom_bond_creation():
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 6, 0, 0, 0)
    molecular.add_atom(molecule, 6, 1, 0, 0)
    molecular.add_bond(molecule, 0, 1, 1)
    npt.assert_equal(molecular.get_total_num_bonds(molecule), 1)
    npt.assert_equal(molecular.get_total_num_atoms(molecule), 2)


def test_atomic_number():
    # Testing atomic number get/set functions
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 4, 0, 0, 0)

    # Testing get_atomic_number
    npt.assert_equal(molecular.get_atomic_number(molecule, 0), 4)

    # Testing set_atomic_number
    molecular.set_atomic_number(molecule, 0, 6)
    npt.assert_equal(molecular.get_atomic_number(molecule, 0), 6)


def test_atomic_position():
    # Testing atomic position get/set functions
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 4, 0, 0, 0)

    # Testing get_atomic_position
    npt.assert_array_almost_equal(molecular.get_atomic_position(molecule, 0),
                                  np.array([0, 0, 0]))

    # Testing set_atomic_number
    molecular.set_atomic_position(molecule, 0, 1, 1, 1)
    npt.assert_array_almost_equal(molecular.get_atomic_position(molecule, 0),
                                  np.array([1, 1, 1]))


def test_bond_order():
    # Testing bond order get/set functions

    # Testing get_bond_order
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 6, 0, 0, 0)
    molecular.add_atom(molecule, 6, 1, 0, 0)
    molecular.add_bond(molecule, 0, 1, 3)
    npt.assert_equal(molecular.get_bond_order(molecule, 0), 3)

    # Testing set_bond_order
    molecular.set_bond_order(molecule, 0, 2)
    npt.assert_equal(molecular.get_bond_order(molecule, 0), 2)

    # Testing get_bond_orders_array
    npt.assert_array_almost_equal(molecular.get_bond_orders_array(molecule),
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
    scene = window.Scene()
    for i, colormode in enumerate(colormodes):
        test_actor = molecular.sphere_rep_actor(molecule, colormode)

        scene.add(test_actor)
        scene.reset_camera()
        scene.reset_clipping_range()

        if interactive:
            window.show(scene)

        npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr, colors=colors[i])
        npt.assert_equal(report.objects, 1)
        scene.clear()


def test_bstick_rep_actor(interactive=False):
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 6, 0, 0, 0)
    molecular.add_atom(molecule, 6, 2, 0, 0)
    molecular.add_bond(molecule, 0, 1, 1)
    colormodes = ['discrete', 'single']
    atom_scale_factor = [0.3, 0.4]
    bond_thickness = [1, 1.2]
    multiple_bonds = ['On', 'Off']
    table = molecular.PeriodicTable()
    colors = np.array([[table.atom_color(6)],
                       [[150/255, 150/255, 150/255],
                        [50/255, 50/255, 50/255]]], dtype=object)
    scene = window.Scene()
    for i, colormode in enumerate(colormodes):
        test_actor = molecular.bstick_rep_actor(molecule, colormode,
                                                atom_scale_factor[i],
                                                bond_thickness[i],
                                                multiple_bonds[i])
        scene.add(test_actor)
        scene.reset_camera()
        scene.reset_clipping_range()

        if interactive:
            window.show(scene)

        npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr, colors=colors[i])
        npt.assert_equal(report.objects, 1)
        scene.clear()


def test_stick_rep_actor(interactive=False):
    molecule = molecular.Molecule()
    molecular.add_atom(molecule, 6, 0, 0, 0)
    molecular.add_atom(molecule, 6, 2, 0, 0)
    molecular.add_bond(molecule, 0, 1, 1)
    colormodes = ['discrete', 'single']
    bond_thickness = [1, 1.2]
    table = molecular.PeriodicTable()
    colors = np.array([[table.atom_color(6)],
                       [[150/255, 150/255, 150/255],
                        [50/255, 50/255, 50/255]]], dtype=object)
    scene = window.Scene()
    for i, colormode in enumerate(colormodes):
        test_actor = molecular.stick_rep_actor(molecule, colormode,
                                               bond_thickness[i])
        scene.add(test_actor)
        scene.reset_camera()
        scene.reset_clipping_range()

        if interactive:
            window.show(scene)

        npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr, colors=colors[i])
        npt.assert_equal(report.objects, 1)
        scene.clear()


def test_ribbon_rep_actor(interactive=False):

    # Testing if heteroatoms are rendered properly
    scene = window.Scene()
    elements, atom_coords, atom_types, model, residue_seq, chain, is_hetatm, \
        sheet, helix = get_default_molecular_info(True)
    molecule = molecular.Molecule(elements, atom_coords, atom_types, model,
                                  residue_seq, chain, is_hetatm, sheet, helix)
    test_actor = molecular.ribbon_rep_actor(molecule)
    scene.add(test_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene)

    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

    colors = np.array([[150/255, 150/255, 150/255], [50/255, 50/255, 50/255]])
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 1)
    scene.clear()

    # Testing if helices and sheets are rendered properly
    atom_coords = np.array([[31.726, 105.084,  71.456],
                            [31.477, 105.680,  70.156],
                            [32.599, 106.655,  69.845],
                            [32.634, 107.264,  68.776],
                            [30.135, 106.407,  70.163],
                            [29.053, 105.662,  70.913],
                            [28.118, 106.591,  71.657],
                            [28.461, 107.741,  71.938],
                            [26.928, 106.097,  71.983],
                            [33.507, 106.802,  70.804],
                            [34.635, 107.689,  70.622],
                            [35.687, 107.018,  69.765],
                            [36.530, 107.689,  69.174],
                            [35.631, 105.690,  69.688],
                            [36.594, 104.921,  68.903],
                            [36.061, 104.498,  67.534],
                            [36.601, 103.580,  66.916],
                            [37.047, 103.645,  69.660],
                            [35.907, 102.828,  69.957],
                            [37.751, 104.014,  70.958]])
    elements = np.array([7, 6, 6, 8, 6, 6, 6, 8, 7, 7, 6, 6, 8, 7, 6, 6, 8, 6,
                         8, 6])
    atom_types = np.array(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2',
                           'N', 'CA', 'C', 'O', 'N', 'CA', 'C', 'O', 'CB',
                           'OG1', 'OG2'])
    model = np.ones(20)
    chain = np.ones(20)*65
    residue_seq = np.ones(20)
    residue_seq[9:13] = 2
    residue_seq[13:] = 3
    residue_seq[6] = 4
    is_hetatm = np.zeros(20, dtype=bool)
    secondary_structure = np.array([[65, 1, 65, 3]])
    colors = np.array([[240/255, 0, 128/255], [1, 1, 0]])
    for i, color in enumerate(colors):
        if i:
            helix = []
            sheet = secondary_structure
        else:
            helix = secondary_structure
            sheet = []
        molecule = molecular.Molecule(elements, atom_coords, atom_types, model,
                                      residue_seq, chain, is_hetatm, sheet,
                                      helix)
        test_actor = molecular.ribbon_rep_actor(molecule)
        scene.set_camera((28, 113, 74), (34, 106, 70), (-0.37, 0.29, -0.88))
        scene.add(test_actor)
        scene.reset_camera()
        scene.reset_clipping_range()

        if interactive:
            window.show(scene)
        npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr, colors=[color])
        npt.assert_equal(report.objects, 1)
        scene.clear()
