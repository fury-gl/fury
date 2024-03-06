"""Module that provides molecular visualization tools."""

import warnings

import numpy as np

from fury.actor import streamtube
from fury.lib import (
    VTK_FLOAT,
    VTK_ID_TYPE,
    VTK_UNSIGNED_CHAR,
    VTK_UNSIGNED_INT,
    VTK_UNSIGNED_SHORT,
    Actor,
    DataSetAttributes,
)
from fury.lib import Molecule as Mol
from fury.lib import (
    OpenGLMoleculeMapper,
    PeriodicTable,
    PolyData,
    PolyDataMapper,
    ProteinRibbonFilter,
    SimpleBondPerceiver,
    StringArray,
)
from fury.lib import numpy_support as nps
from fury.utils import numpy_to_vtk_points


class Molecule(Mol):
    """Your molecule class.

    An object that is used to create molecules and store molecular data (e.g.
    coordinate and bonding data).
    This is a more pythonic version of ``Molecule``.
    """

    def __init__(
        self,
        atomic_numbers=None,
        coords=None,
        atom_names=None,
        model=None,
        residue_seq=None,
        chain=None,
        sheet=None,
        helix=None,
        is_hetatm=None,
    ):
        """Send the atomic data to the molecule.

        Parameters
        ----------
        atomic_numbers : ndarray of integers, optional
            The shape of the array must be (N, ) where N is the total number of
            atoms present in the molecule.
            Array having atomic number corresponding to each atom of the
            molecule.
        coords : ndarray of floats, optional
            The shape of the array must be (N, 3) where N is the total number
            of atoms present in the molecule.
            Array having coordinates corresponding to each atom of the
            molecule.
        atom_names : ndarray of strings, optional
            The shape of the array must be (N, ) where N is the total number of
            atoms present in the molecule.
            Array having the names of atoms.
        model : ndarray of integers, optional
            The shape of the array must be (N, ) where N is the total number of
            atoms present in the molecule.
            Array having the model number corresponding to each atom.
        residue_seq : ndarray of integers, optional
            The shape of the array must be (N, ) where N is the total number of
            atoms present in the molecule.
            Array having the residue sequence number corresponding to each atom
            of the molecule.
        chain : ndarray of integers, optional
            The shape of the array must be (N, ) where N is the total number of
            atoms present in the molecule.
            Array having the chain number corresponding to each atom.
        sheet : ndarray of integers, optional
            The shape of the array must be (S, 4) where S is the total number
            of sheets present in the molecule.
            Array containing information about sheets present in the molecule.
        helix : ndarray of integers, optional
            The shape of the array must be (H, 4) where H is the total number
            of helices present in the molecule.
            Array containing information about helices present in the molecule.
        is_hetatm : ndarray of bools, optional
            The shape of the array must be (N, ) where N is the total number of
            atoms present in the molecule.
            Array containing a bool value to indicate if an atom is a
            heteroatom.

        """
        if atomic_numbers is None and coords is None:
            self.Initialize()
        elif not isinstance(atomic_numbers, np.ndarray) or not isinstance(
            coords, np.ndarray
        ):
            raise ValueError('atom_types and coords must be numpy arrays.')
        elif len(atomic_numbers) == len(coords):
            self.atom_names = atom_names
            self.model = model
            self.residue_seq = residue_seq
            self.chain = chain
            self.sheet = sheet
            self.helix = helix
            self.is_hetatm = is_hetatm
            coords = numpy_to_vtk_points(coords)
            atom_nums = nps.numpy_to_vtk(atomic_numbers, array_type=VTK_UNSIGNED_SHORT)
            atom_nums.SetName('Atomic Numbers')
            fieldData = DataSetAttributes()
            fieldData.AddArray(atom_nums)
            self.Initialize(coords, fieldData)
        else:
            n1 = len(coords)
            n2 = len(atomic_numbers)
            raise ValueError(
                'Mismatch in length of atomic_numbers({0}) and '
                'length of atomic_coords({1}).'.format(n1, n2)
            )

    @property
    def total_num_atoms(self):
        """Return the total number of atoms in a given molecule."""
        return self.GetNumberOfAtoms()

    @property
    def total_num_bonds(self):
        """Return the total number of bonds in a given molecule."""
        return self.GetNumberOfBonds()


def add_atom(molecule, atomic_num, x_coord, y_coord, z_coord):
    """Add atomic data to our molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the atom is to be added.
    atomic_num : int
        Atomic number of the atom.
    x_coord : float
        x-coordinate of the atom.
    y_coord : float
        y-coordinate of the atom.
    z_coord : float
        z-coordinate of the atom.

    """
    molecule.AppendAtom(atomic_num, x_coord, y_coord, z_coord)


def add_bond(molecule, atom1_index, atom2_index, bond_order=1):
    """Add bonding data to our molecule. Establish a bond of type bond_order
    between the atom at atom1_index and the atom at atom2_index.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the bond is to be added.
    atom1_index : int
        Index of the first atom.
    atom2_index : int
        Index of the second atom.
    bond_order : int (optional)
        Bond order (whether it's a single/double/triple bond). Default: 1

    Notes
    -----
    Ensure that the total number of bonds between two atoms doesn't exceed 3.
    Calling ``add_bond`` to add bonds between atoms that already have a triple
    bond between them leads to erratic behavior and must be avoided.

    """
    molecule.AppendBond(atom1_index, atom2_index, bond_order)


def get_atomic_number(molecule, atom_index):
    """Get the atomic number of an atom for a specified index.

    Returns the atomic number of the atom present at index atom_index.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the atom belongs.
    atom_index : int
        Index of the atom whose atomic number is to be obtained.

    """
    return molecule.GetAtomAtomicNumber(atom_index)


def set_atomic_number(molecule, atom_index, atomic_num):
    """Set the atomic number of an atom for a specified index.

    Assign atomic_num as the atomic number of the atom present at
    atom_index.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the atom belongs.
    atom_index : int
        Index of the atom to whom the atomic number is to be assigned.
    atom_num : int
        Atomic number to be assigned to the atom.

    """
    molecule.SetAtomAtomicNumber(atom_index, atomic_num)


def get_atomic_position(molecule, atom_index):
    """Get the atomic coordinates of an atom for a specified index.

    Returns the atomic coordinates of the atom present at index atom_index.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the atom belongs.
    atom_index : int
        Index of the atom whose atomic coordinates are to be obtained.

    """
    return molecule.GetAtomPosition(atom_index)


def set_atomic_position(molecule, atom_index, x_coord, y_coord, z_coord):
    """Set the atomic coordinates of an atom for a specified index.

    Assign atom_coordinate to the coordinates of the atom present at
    atom_index.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the atom belongs.
    atom_index : int
        Index of the atom to which the coordinates are to be assigned.
    x_coord : float
        x-coordinate of the atom.
    y_coord : float
        y-coordinate of the atom.
    z_coord : float
        z-coordinate of the atom.

    """
    molecule.SetAtomPosition(atom_index, x_coord, y_coord, z_coord)


def get_bond_order(molecule, bond_index):
    """Get the order of bond for a specified index.

    Returns the order of bond (whether it's a single/double/triple bond)
    present at bond_index.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the bond belongs.
    bond_index : int
        Index of the bond whose order is to be obtained.

    """
    return molecule.GetBondOrder(bond_index)


def set_bond_order(molecule, bond_index, bond_order):
    """Set the bond order of a bond for a specified index.

    Assign bond_order (whether it's a single/double/triple bond) to the bond
    present at the bond_index.

    Parameters
    ----------
    molecule : Molecule
        The molecule to which the bond belongs.
    bond_index : int
        Index of the bond whose order is to be assigned.
    bond_order : int
        Bond order (whether it's a single/double/triple bond).

    """
    return molecule.SetBondOrder(bond_index, bond_order)


def get_all_atomic_numbers(molecule):
    """Return an array of atomic numbers corresponding to the atoms
    present in a given molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule whose atomic number array is to be obtained.

    """
    return nps.vtk_to_numpy(molecule.GetAtomicNumberArray())


def get_all_bond_orders(molecule):
    """Return an array of integers containing the bond orders (single/double/
    triple) corresponding to the bonds present in the molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule whose bond types array is to be obtained.

    """
    return nps.vtk_to_numpy(molecule.GetBondOrdersArray())


def get_all_atomic_positions(molecule):
    """Return an array of atomic coordinates corresponding to the atoms
    present in the molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule whose atomic position array is to be obtained.

    """
    return nps.vtk_to_numpy(molecule.GetAtomicPositionArray().GetData())


def deep_copy_molecule(molecule1, molecule2):
    """Deep copies the atomic information (atoms and bonds) from molecule2 into
    molecule1.

    Parameters
    ----------
    molecule1 : Molecule
        The molecule to which the atomic information is copied.
    molecule2 : Molecule
        The molecule from which the atomic information is copied.

    """
    molecule1.DeepCopyStructure(molecule2)


def compute_bonding(molecule):
    """Uses `vtkSimpleBondPerceiver` to generate bonding information for a
    molecule.
    `vtkSimpleBondPerceiver` performs a simple check of all interatomic
    distances and adds a single bond between atoms that are reasonably
    close. If the interatomic distance is less than the sum of the two
    atom's covalent radii plus a tolerance, a single bond is added.

    Parameters
    ----------
    molecule : Molecule
        The molecule for which bonding information is to be generated.

    Notes
    -----
    This algorithm does not consider valences, hybridization, aromaticity,
    or anything other than atomic separations. It will not produce anything
    other than single bonds.

    """
    bonder = SimpleBondPerceiver()
    bonder.SetInputData(molecule)
    bonder.SetTolerance(0.1)
    bonder.Update()
    deep_copy_molecule(molecule, bonder.GetOutput())


class PTable(PeriodicTable):
    """A class to obtain properties of elements (eg: Covalent Radius,
    Van Der Waals Radius, Symbol etc.).

    This is a more pythonic version of ``vtkPeriodicTable`` providing simple
    methods to access atomic properties. It provides access to essential
    functionality available in ``vtkPeriodicTable``. An object of this class
    provides access to atomic information sourced from Blue Obelisk Data
    Repository.
    """

    def atomic_symbol(self, atomic_number):
        """Given an atomic number, returns the symbol associated with the
        element.

        Parameters
        ----------
        atomic_number : int
            Atomic number of the element whose symbol is to be obtained.

        """
        return self.GetSymbol(atomic_number)

    def element_name(self, atomic_number):
        """Given an atomic number, returns the name of the element.

        Parameters
        ----------
        atomic_number : int
            Atomic number of the element whose name is to be obtained.

        """
        return self.GetElementName(atomic_number)

    def atomic_number(self, element_name):
        """Given a case-insensitive string that contains the symbol or name of
        an element, return the corresponding atomic number.

        Parameters
        ----------
        element_name : string
            Name of the element whose atomic number is to be obtained.

        """
        return self.GetAtomicNumber(element_name)

    def atomic_radius(self, atomic_number, radius_type='VDW'):
        """Given an atomic number, return either the covalent radius of the
        atom (in Å) or return the Van Der Waals radius (in Å) of the atom
        depending on radius_type.

        Parameters
        ----------
        atomic_number : int
            Atomic number of the element whose atomic radius is to be obtained.
        radius_type : string
            Type of atomic radius to be obtained. Two valid choices:

            * 'VDW' : for Van der Waals radius of the atom
            * 'Covalent' : for covalent radius of the atom

            Default: 'VDW'

        """
        radius_type = radius_type.lower()
        if radius_type == 'vdw':
            return self.GetVDWRadius(atomic_number)
        elif radius_type == 'covalent':
            return self.GetCovalentRadius(atomic_number)
        else:
            raise ValueError(
                'Incorrect radius_type specified. Please choose'
                ' from "VDW" or "Covalent".'
            )

    def atom_color(self, atomic_number):
        """Given an atomic number, return the RGB tuple associated with that
        element (CPK coloring convention) provided by the Blue Obelisk Data
        Repository.

        Parameters
        ----------
        atomicNumber : int
            Atomic number of the element whose RGB tuple is to be obtained.

        """
        rgb = np.array(self.GetDefaultRGBTuple(atomic_number))
        return rgb


def sphere_cpk(molecule, colormode='discrete'):
    """Create an actor for sphere molecular representation. It's also referred
    to as CPK model and space-filling model.

    Parameters
    ----------
    molecule : Molecule
        The molecule to be rendered.
    colormode : string, optional
        Set the colormode for coloring the atoms. Two valid color modes:

        * 'discrete': Atoms are colored using CPK coloring convention.
        * 'single': All atoms are colored with same color (grey). RGB tuple
          used for coloring the atoms when 'single' colormode is selected:
          (150, 150, 150).

        Default: 'discrete'

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the space filling representation of the
        molecule to be visualized.

    References
    ----------
    Corey R.B.; Pauling L. Molecular Models of Amino Acids,
    Peptides, and Proteins
    `Review of Scientific Instruments 1953, 24 (8), 621-627.
    <https://doi.org/10.1063/1.1770803>`_

    """
    colormode = colormode.lower()
    msp_mapper = OpenGLMoleculeMapper()
    msp_mapper.SetInputData(molecule)
    msp_mapper.SetRenderAtoms(True)
    msp_mapper.SetRenderBonds(False)
    msp_mapper.SetAtomicRadiusTypeToVDWRadius()
    msp_mapper.SetAtomicRadiusScaleFactor(1)
    if colormode == 'discrete':
        msp_mapper.SetAtomColorMode(1)
    elif colormode == 'single':
        msp_mapper.SetAtomColorMode(0)
    else:
        msp_mapper.SetAtomColorMode(1)
        warnings.warn('Incorrect colormode specified! Using discrete.')

    # To-Do manipulate shading properties to make it look aesthetic
    molecule_actor = Actor()
    molecule_actor.SetMapper(msp_mapper)
    return molecule_actor


def ball_stick(
    molecule,
    colormode='discrete',
    atom_scale_factor=0.3,
    bond_thickness=0.1,
    multiple_bonds=True,
):
    """Create an actor for ball and stick molecular representation.

    Parameters
    ----------
    molecule : Molecule
        The molecule to be rendered.
    colormode : string, optional
        Set the colormode for coloring the atoms. Two valid color modes:

        * 'discrete': Atoms and bonds are colored using CPK coloring
          convention.
        * 'single': All atoms are colored with same color (grey) and all bonds
          are colored with same color (dark grey). RGB tuple used for coloring
          the atoms when 'single' colormode is selected: (150, 150, 150). RGB
          tuple used for coloring the bonds when 'single' colormode is
          selected: (50, 50, 50)

        Default: 'discrete'
    atom_scale_factor : float, optional
        Scaling factor. Default: 0.3
    bond_thickness : float, optional
        Used to manipulate the thickness of bonds (i.e. thickness of tubes
        which are used to render bonds)
        Default: 0.1 (Optimal range: 0.1 - 0.5).
    multiple_bonds : bool, optional
        Set whether multiple tubes will be used to represent multiple
        bonds. If True, multiple bonds (double, triple) will be shown by using
        multiple tubes. If False, all bonds (single, double, triple) will be
        shown as single bonds (i.e. shown using one tube each).
        Default is True.

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the ball and stick representation of the
        molecule to be visualized.

    References
    ----------
    Turner, M. Ball and stick models for organic chemistry
    `J. Chem. Educ. 1971, 48, 6, 407.
    <https://doi.org/10.1021/ed048p407>`_

    """
    if molecule.total_num_bonds == 0:
        raise ValueError(
            'No bonding data available for the molecule! Ball '
            'and stick model cannot be made!'
        )
    colormode = colormode.lower()
    bs_mapper = OpenGLMoleculeMapper()
    bs_mapper.SetInputData(molecule)
    bs_mapper.SetRenderAtoms(True)
    bs_mapper.SetRenderBonds(True)
    bs_mapper.SetBondRadius(bond_thickness)
    bs_mapper.SetAtomicRadiusTypeToVDWRadius()
    bs_mapper.SetAtomicRadiusScaleFactor(atom_scale_factor)
    if multiple_bonds:
        bs_mapper.SetUseMultiCylindersForBonds(1)
    else:
        bs_mapper.SetUseMultiCylindersForBonds(0)
    if colormode == 'discrete':
        bs_mapper.SetAtomColorMode(1)
        bs_mapper.SetBondColorMode(1)
    elif colormode == 'single':
        bs_mapper.SetAtomColorMode(0)
        bs_mapper.SetBondColorMode(0)
    else:
        bs_mapper.SetAtomColorMode(1)
        warnings.warn('Incorrect colormode specified! Using discrete.')
    molecule_actor = Actor()
    molecule_actor.SetMapper(bs_mapper)
    return molecule_actor


def stick(molecule, colormode='discrete', bond_thickness=0.1):
    """Create an actor for stick molecular representation.

    Parameters
    ----------
    molecule : Molecule
        The molecule to be rendered.
    colormode : string, optional
        Set the colormode for coloring the bonds. Two valid color modes:

        * 'discrete': Bonds are colored using CPK coloring convention.
        * 'single': All bonds are colored with the same color (dark grey)
          RGB tuple used for coloring the bonds when 'single' colormode is
          selected: (50, 50, 50)

        Default: 'discrete'

    bond_thickness : float, optional
        Used to manipulate the thickness of bonds (i.e. thickness of tubes
        which are used to render bonds).
        Default: 0.1 (Optimal range: 0.1 - 0.5).

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the stick representation of the molecule to be
        visualized.

    """
    if molecule.total_num_bonds == 0:
        raise ValueError(
            'No bonding data available for the molecule! Stick ' 'model cannot be made!'
        )
    colormode = colormode.lower()
    mst_mapper = OpenGLMoleculeMapper()
    mst_mapper.SetInputData(molecule)
    mst_mapper.SetRenderAtoms(True)
    mst_mapper.SetRenderBonds(True)
    mst_mapper.SetBondRadius(bond_thickness)
    mst_mapper.SetAtomicRadiusTypeToUnitRadius()
    mst_mapper.SetAtomicRadiusScaleFactor(bond_thickness)
    if colormode == 'discrete':
        mst_mapper.SetAtomColorMode(1)
        mst_mapper.SetBondColorMode(1)
    elif colormode == 'single':
        mst_mapper.SetAtomColorMode(0)
        mst_mapper.SetBondColorMode(0)
    else:
        mst_mapper.SetAtomColorMode(1)
        warnings.warn('Incorrect colormode specified! Using discrete.')
    molecule_actor = Actor()
    molecule_actor.SetMapper(mst_mapper)
    return molecule_actor


def ribbon(molecule):
    """Create an actor for ribbon molecular representation.

    Parameters
    ----------
    molecule : Molecule
        The molecule to be rendered.

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the rubbon representation of the molecule to be
        visualized.

    References
    ----------
    Richardson, J.S. The anatomy and taxonomy of protein structure
    `Advances in Protein Chemistry, 1981, 34, 167-339.
    <https://doi.org/10.1016/S0065-3233(08)60520-3>`_

    """
    coords = get_all_atomic_positions(molecule)
    all_atomic_numbers = get_all_atomic_numbers(molecule)
    num_total_atoms = molecule.total_num_atoms
    secondary_structures = np.ones(num_total_atoms)
    for i in range(num_total_atoms):
        secondary_structures[i] = ord('c')
        resi = molecule.residue_seq[i]
        for j, _ in enumerate(molecule.sheet):
            sheet = molecule.sheet[j]
            if molecule.chain[i] != sheet[0] or resi < sheet[1] or resi > sheet[3]:
                continue
            secondary_structures[i] = ord('s')

        for j, _ in enumerate(molecule.helix):
            helix = molecule.helix[j]
            if molecule.chain[i] != helix[0] or resi < helix[1] or resi > helix[3]:
                continue
            secondary_structures[i] = ord('h')

    output = PolyData()

    # for atomic numbers
    atomic_num_arr = nps.numpy_to_vtk(
        num_array=all_atomic_numbers, deep=True, array_type=VTK_ID_TYPE
    )

    # setting the array name to atom_type as vtkProteinRibbonFilter requires
    # the array to be named atom_type
    atomic_num_arr.SetName('atom_type')

    output.GetPointData().AddArray(atomic_num_arr)

    # for atom names
    atom_names = StringArray()

    # setting the array name to atom_types as vtkProteinRibbonFilter requires
    # the array to be named atom_types
    atom_names.SetName('atom_types')
    atom_names.SetNumberOfTuples(num_total_atoms)
    for i in range(num_total_atoms):
        atom_names.SetValue(i, molecule.atom_names[i])

    output.GetPointData().AddArray(atom_names)

    # for residue sequences
    residue_seq = nps.numpy_to_vtk(
        num_array=molecule.residue_seq, deep=True, array_type=VTK_ID_TYPE
    )
    residue_seq.SetName('residue')
    output.GetPointData().AddArray(residue_seq)

    # for chain
    chain = nps.numpy_to_vtk(
        num_array=molecule.chain, deep=True, array_type=VTK_UNSIGNED_CHAR
    )
    chain.SetName('chain')
    output.GetPointData().AddArray(chain)

    # for secondary structures
    s_s = nps.numpy_to_vtk(
        num_array=secondary_structures, deep=True, array_type=VTK_UNSIGNED_CHAR
    )
    s_s.SetName('secondary_structures')
    output.GetPointData().AddArray(s_s)

    # for secondary structures begin
    newarr = np.ones(num_total_atoms)
    s_sb = nps.numpy_to_vtk(num_array=newarr, deep=True, array_type=VTK_UNSIGNED_CHAR)
    s_sb.SetName('secondary_structures_begin')
    output.GetPointData().AddArray(s_sb)

    # for secondary structures end
    newarr = np.ones(num_total_atoms)
    s_se = nps.numpy_to_vtk(num_array=newarr, deep=True, array_type=VTK_UNSIGNED_CHAR)
    s_se.SetName('secondary_structures_end')
    output.GetPointData().AddArray(s_se)

    # for is_hetatm
    is_hetatm = nps.numpy_to_vtk(
        num_array=molecule.is_hetatm, deep=True, array_type=VTK_UNSIGNED_CHAR
    )
    is_hetatm.SetName('ishetatm')
    output.GetPointData().AddArray(is_hetatm)

    # for model
    model = nps.numpy_to_vtk(
        num_array=molecule.model, deep=True, array_type=VTK_UNSIGNED_INT
    )
    model.SetName('model')
    output.GetPointData().AddArray(model)

    table = PTable()

    # for colors and radii of hetero-atoms
    radii = np.ones((num_total_atoms, 3))
    rgb = np.ones((num_total_atoms, 3))

    for i in range(num_total_atoms):
        radii[i] = np.repeat(table.atomic_radius(all_atomic_numbers[i], 'VDW'), 3)
        rgb[i] = table.atom_color(all_atomic_numbers[i])

    Rgb = nps.numpy_to_vtk(num_array=rgb, deep=True, array_type=VTK_UNSIGNED_CHAR)
    Rgb.SetName('rgb_colors')
    output.GetPointData().SetScalars(Rgb)

    Radii = nps.numpy_to_vtk(num_array=radii, deep=True, array_type=VTK_FLOAT)
    Radii.SetName('radius')
    output.GetPointData().SetVectors(Radii)

    # setting the coordinates
    points = numpy_to_vtk_points(coords)
    output.SetPoints(points)

    ribbonFilter = ProteinRibbonFilter()
    ribbonFilter.SetInputData(output)
    ribbonFilter.SetCoilWidth(0.2)
    ribbonFilter.SetDrawSmallMoleculesAsSpheres(0)
    mapper = PolyDataMapper()
    mapper.SetInputConnection(ribbonFilter.GetOutputPort())
    molecule_actor = Actor()
    molecule_actor.SetMapper(mapper)
    return molecule_actor


def bounding_box(molecule, colors=(1, 1, 1), linewidth=0.3):
    """Create a bounding box for a molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule around which the bounding box is to be created.
    colors : tuple (3,) or ndarray of shape (3,), optional
        Color of the bounding box. Default: (1, 1, 1)
    linewidth: float, optional
        Thickness of tubes used to compose bounding box. Default: 0.3

    Returns
    -------
    bbox_actor : vtkActor
        Actor created to serve as a bounding box for a given molecule.

    """
    pts = numpy_to_vtk_points(get_all_atomic_positions(molecule))
    min_x, max_x, min_y, max_y, min_z, max_z = pts.GetBounds()

    lines = np.array(
        [
            [[min_x, min_y, min_z], [min_x, min_y, max_z]],
            [[min_x, max_y, min_z], [min_x, max_y, max_z]],
            [[max_x, min_y, min_z], [max_x, min_y, max_z]],
            [[max_x, max_y, min_z], [max_x, max_y, max_z]],
            [[min_x, min_y, min_z], [max_x, min_y, min_z]],
            [[min_x, max_y, min_z], [max_x, max_y, min_z]],
            [[min_x, max_y, max_z], [max_x, max_y, max_z]],
            [[min_x, min_y, max_z], [max_x, min_y, max_z]],
            [[min_x, min_y, min_z], [min_x, max_y, min_z]],
            [[max_x, min_y, min_z], [max_x, max_y, min_z]],
            [[min_x, min_y, max_z], [min_x, max_y, max_z]],
            [[max_x, min_y, max_z], [max_x, max_y, max_z]],
        ]
    )

    return streamtube(lines, colors=colors, linewidth=linewidth)
