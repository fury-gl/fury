import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
from fury.utils import numpy_to_vtk_points


class Molecule(vtk.vtkMolecule):
    """Your molecule class.

    An object that is used to create molecules and store molecular data (e.g.
    coordinate and bonding data).
    This is a more pythonic version of ``vtkMolecule``.
    """

    def __init__(self, elements=None, coords=None, atom_types=None, model=None,
                 residue_seq=None, chain=None, sheet=None, helix=None,
                 is_hetatm=None):
        """Send the atomic data to the molecule.

        Parameters
        ----------
        elements : ndarray of integers, shape (N, ) where N is the total number
            of atoms present in the molecule.
            Array having atomic number corresponding to each atom of the
            molecule.
        coords : ndarray of floats, shape (N, 3) where N is the total number
            of atoms present in the molecule.
            Array having coordinates corresponding to each atom of the
            molecule.
        atom_types : ndarray of strings, shape (N, ) where N is the total
            number of atoms present in the molecule.
            Array having the name of type of atom.
        model : ndarray of integers, shape (N, ) where N is the total number of
            atoms present in the molecule.
            Array having the model number corresponding to each atom.
        residue_seq : ndarray of integers, shape (N, ) where N is the total
            number of atoms present in the molecule.
            Array having the residue sequence number corresponding to each atom
            of the molecule.
        chain : ndarray of integers, shape (N, ) where N is the total number of
            atoms present in the molecule.
            Array having the chain number corresponding to each atom.
        sheet : ndarray of integers, shape (S, 4) where S is the total number
            of sheets present in the molecule.
            Array containing information about sheets present in the molecule.
        helix : ndarray of integers, shape (H, 4) where H is the total number
            of helices present in the molecule.
            Array containing information about helices present in the molecule.
        """
        if elements.any() and coords.any() and len(elements) == len(coords):
        # To-do: raise an error if the length of elements and coords are unequal
            self.atom_types = atom_types
            self.model = model
            self.residue_seq = residue_seq
            self.chain = chain
            self.sheet = sheet
            self.helix = helix
            self.is_hetatm = is_hetatm
            coords = numpy_to_vtk_points(coords)
            atom_nums = numpy_to_vtk(elements,
                                     array_type=vtk.VTK_UNSIGNED_SHORT)
            atom_nums.SetName("Atomic Numbers")
            fieldData = vtk.vtkDataSetAttributes()
            fieldData.AddArray(atom_nums)
            self.Initialize(coords, fieldData)


def add_atom(molecule, atomic_num, x_coord, y_coord, z_coord):
    """Add atomic data to our molecule.

    Parameters
    ----------
    molecule : Molecule() object
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


def add_bond(molecule, atom1_index, atom2_index, bond_type=1):
    """Add bonding data to our molecule. Establish a bond of type bond_type
    between the atom at atom1_index and the atom at atom2_index.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule to which the bond is to be added.
    atom1_index : int
        Index of the first atom.
    atom2_index : int
        Index of the second atom.
    bond_type : int (optional)
        Type of bond (single/double/triple). Default: 1
    """
    molecule.AppendBond(atom1_index, atom2_index, bond_type)


def get_total_num_atoms(molecule):
    """Returns the total number of atoms in a given molecule.

    Parameters
    ----------
    molecule : Molecule() object
    """
    return molecule.GetNumberOfAtoms()


def get_total_num_bonds(molecule):
    """Returns the total number of bonds in a given molecule.

    Parameters
    ----------
    molecule : Molecule() object
    """
    return molecule.GetNumberOfBonds()


def get_atomic_number(molecule, atom_index):
    """Get the atomic number of an atom for a specified index.

    Returns the atomic number of the atom present at index atom_index.

    Parameters
    ----------
    molecule : Molecule() object
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
    molecule : Molecule() object
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
    molecule : Molecule() object
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
    molecule : Molecule() object
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


def get_bond_type(molecule, bond_index):
    """Get the type of bond for a specified index.

    Returns the type of bond (whether it's a single/double/triple bond)
    present at bond_index.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule to which the bond belongs.
    bond_index : int
        Index of the bond whose type is to be obtained.
    """
    return molecule.GetBondOrder(bond_index)


def set_bond_type(molecule, bond_index, bond_type):
    """Set the bond type of a bond for a specified index.

    Assign bond_type (whether it's a single/double/triple bond) to the bond
    present at the bond_index.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule to which the bond belongs.
    bond_index : int
        Index of the atom to which the coordinates are to be assigned.
    bond_type : int
        Type of the bond (single/double/triple).
    """
    return molecule.SetBondOrder(bond_index, bond_type)


def get_atomic_number_array(molecule):
    """Returns an array of atomic numbers corresponding to the atoms
    present in a given molecule.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule whose atomic number array is to be obtained.
    """
    return vtk_to_numpy(molecule.GetAtomicNumberArray())


def get_bond_types_array(molecule):
    """Returns an array containing the types of the bond (single/double/
    triple) corresponding to the bonds present in the molecule.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule whose bond types array is to be obtained.
    """
    return vtk_to_numpy(molecule.GetBondOrdersArray())


def get_atomic_position_array(molecule):
    """Returns an array of atomic coordinates corresponding to the atoms
    present in the molecule.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule whose atomic position array is to be obtained.
    """
    return vtk_to_numpy(molecule.GetAtomicPositionArray().GetData())


def deep_copy(molecule1, molecule2):
    """
    Deep copies the atomic information (atoms and bonds) from molecule2 into
    molecule1.

    Parameters
    ----------
    molecule1 : Molecule() object
        The molecule to which the atomic information is copied.
    molecule2 : Molecule() object
        The molecule from which the atomic information is copied.
    """
    molecule1.DeepCopyStructure(molecule2)


def compute_bonding(molecule):
    """
    Uses vtkSimpleBondPerceiver to generate bonding information for a
    molecule.
    vtkSimpleBondPerceiver performs a simple check of all interatomic
    distances and adds a single bond between atoms that are reasonably
    close. If the interatomic distance is less than the sum of the two
    atom's covalent radii plus a tolerance, a single bond is added.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule for which bonding information is to be generated.

    Notes
    -----
    This algorithm does not consider valences, hybridization, aromaticity,
    or anything other than atomic separations. It will not produce anything
    other than single bonds.
    """
    bonder = vtk.vtkSimpleBondPerceiver()
    bonder.SetInputData(molecule)
    bonder.SetTolerance(0.1)
    bonder.Update()
    deep_copy(molecule, bonder.GetOutput())


class PeriodicTable(vtk.vtkPeriodicTable):
    """ A class to obtain properties of elements (eg: Covalent Radius,
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

    def atomic_radius(self, atomic_number, radius_type):
        """Given an atomic number, return either the covalent radius of the
        atom or return the Van Der Waals radius of the atom depending on
        radius_type.

        Parameters
        ----------
        atomic_number : int
            Atomic number of the element whose covalent radius is to be
            obtained.
        radius_type : string
            Two valid choices -
            * 'VDW' : for Van Der Waals radius of the atom
            * 'Covalent' : for covalent radius of the atom
        """
        if radius_type == 'VDW':
            return self.GetVDWRadius(atomic_number)
        elif radius_type == 'Covalent':
            return self.GetCovalentRadius(atomic_number)

    def atom_color(self, atomic_number):
        """Given an atomic number, return the RGB tuple associated with that
        element (CPK coloring convention) provided by the Blue Obelisk Data
        Repository.

        Parameters
        ----------
        atomicNumber : int
            Atomic number of the element whose RGB tuple is to be obtained.
        """
        return self.GetDefaultRGBTuple(atomic_number)


def make_molecularviz_aesthetic(molecule_actor):
    """Manipulating the lighting to make the molecular visualization
    aesthetically pleasant to see.

    Parameters
    ----------
    molecule_actor : vtkActor
        Actor that represents the molecule to be visualized.
    """
    molecule_actor.GetProperty().SetDiffuse(1)
    molecule_actor.GetProperty().SetSpecular(1)
    molecule_actor.GetProperty().SetAmbient(0.3)
    molecule_actor.GetProperty().SetSpecularPower(100.0)


def sphere_rep_actor(molecule, colormode='discrete'):
    """Create an actor for sphere molecular representation. It's also referred
    to as CPK model and space-filling model.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule to be rendered.
    colormode : string
        Set the colormode for coloring the atoms. Two valid color modes -
        * 'discrete': Atoms are colored using CPK coloring convention.
        * 'single': All atoms are colored with same color(grey).

        RGB tuple used for coloring the atoms when 'single' colormode is
        selected: (150, 150, 150)
        Default is 'discrete'.

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the space filling representation of the
        molecule to be visualized.
    """
    msp_mapper = vtk.vtkOpenGLMoleculeMapper()
    msp_mapper.SetInputData(molecule)
    msp_mapper.SetRenderAtoms(True)
    msp_mapper.SetRenderBonds(False)
    msp_mapper.SetAtomicRadiusTypeToVDWRadius()
    msp_mapper.SetAtomicRadiusScaleFactor(1)
    if colormode == 'discrete':
        msp_mapper.SetAtomColorMode(1)
    elif colormode == 'single':
        msp_mapper.SetAtomColorMode(0)

    molecule_actor = vtk.vtkActor()
    molecule_actor.SetMapper(msp_mapper)
    make_molecularviz_aesthetic(molecule_actor)
    return molecule_actor


def bstick_rep_actor(molecule, colormode='discrete',
                     atom_scale_factor=0.3, bond_thickness=1,
                     multiple_bonds='On'):
    """Create an actor for ball and stick molecular representation.

    Parameters
    ----------
    molecule : Molecule() object
        The molecule to be rendered.
    colormode : string
        Set the colormode for coloring the atoms. Two valid color modes -
        * 'discrete': Atoms and bonds are colored using CPK coloring
          convention.
        * 'single': All atoms are colored with same color(grey) and all bonds
          are colored with same color(dark grey).
        * RGB tuple used for coloring the atoms when 'single' colormode is
        selected: (150, 150, 150)
        * RGB tuple used for coloring the bonds when 'single' colormode is
        selected: (50, 50, 50)
        * Default is 'discrete'.

    atom_scale_factor : float
        Scaling factor colormode='discrete',
                               atom_scale_factor=0.3, bond_thickness=1,
                               multipleBoto be applied to the atoms.
        Default is 0.3.
    bond_thickness : float
        Used to manipulate the thickness of bonds (i.e. thickness of tubes
        which are used to render bonds)
        Default is 1.
    multipleBonds : string
        Set whether multiple tubes will be used to represent multiple
        bonds. Two valid choices -
        * 'On': multiple bonds (double, triple) will be shown by using
          multiple tubes.
        * 'Off': all bonds (single, double, triple) will be shown as single
          bonds (i.e shown using one tube each).
        Default is 'On'.

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the ball and stick representation of the
        molecule to be visualized.
    """
    bs_mapper = vtk.vtkOpenGLMoleculeMapper()
    bs_mapper.SetInputData(molecule)
    bs_mapper.SetRenderAtoms(True)
    bs_mapper.SetRenderBonds(True)
    bs_mapper.SetBondRadius(bond_thickness/10)
    bs_mapper.SetAtomicRadiusTypeToVDWRadius()
    bs_mapper.SetAtomicRadiusScaleFactor(atom_scale_factor)
    if multiple_bonds == 'On':
        bs_mapper.SetUseMultiCylindersForBonds(1)
    elif multiple_bonds == 'Off':
        bs_mapper.SetUseMultiCylindersForBonds(0)
    if colormode == 'discrete':
        bs_mapper.SetAtomColorMode(1)
        bs_mapper.SetBondColorMode(1)
    elif colormode == 'single':
        bs_mapper.SetAtomColorMode(0)
        bs_mapper.SetBondColorMode(0)

    molecule_actor = vtk.vtkActor()
    molecule_actor.SetMapper(bs_mapper)
    make_molecularviz_aesthetic(molecule_actor)
    return molecule_actor


def stick_rep_actor(molecule, colormode='discrete',
                    bond_thickness=1):
    """Create an actor for stick molecular representation.

    Parameters
    ----------
    molecule : Molecule object
        The molecule to be rendered.
    colormode : string
        Set the colormode for coloring the bonds. Two valid color modes -
        * 'discrete': Bonds are colored using CPK coloring convention.
        * 'single': All bonds are colored with the same color (dark grey)
        RGB tuple used for coloring the bonds when 'single' colormode is
        selected: (50, 50, 50)
        Default is 'discrete'.

    bond_thickness : float
        Used to manipulate the thickness of bonds (i.e. thickness of tubes
        which are used to render bonds).
        Default is 1.

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the stick representation of the molecule to be
        visualized.
    """
    mst_mapper = vtk.vtkOpenGLMoleculeMapper()
    mst_mapper.SetInputData(molecule)
    mst_mapper.SetRenderAtoms(True)
    mst_mapper.SetRenderBonds(True)
    mst_mapper.SetBondRadius(bond_thickness/10)
    mst_mapper.SetAtomicRadiusTypeToUnitRadius()
    mst_mapper.SetAtomicRadiusScaleFactor(bond_thickness/10)
    if colormode == 'discrete':
        mst_mapper.SetAtomColorMode(1)
        mst_mapper.SetBondColorMode(1)
    elif colormode == 'single':
        mst_mapper.SetAtomColorMode(0)
        mst_mapper.SetBondColorMode(0)
    molecule_actor = vtk.vtkActor()
    molecule_actor.SetMapper(mst_mapper)
    make_molecularviz_aesthetic(molecule_actor)
    return molecule_actor


def ribbon_rep_actor(molecule):
    """Create an actor for ribbon molecular representation.

    Parameters
    ----------
    molecule : Molecule object
        The molecule to be rendered.

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the rubbon representation of the molecule to be
        visualized.
    """
    coords = get_atomic_position_array(molecule)
    elements = get_atomic_number_array(molecule)
    num_total_atoms = len(coords)
    SecondaryStructures = np.ones(num_total_atoms)
    for i in range(num_total_atoms):
        SecondaryStructures[i] = ord('c')
        resi = molecule.residue_seq[i]
        for j in range(len(molecule.sheet)):
            sheet = molecule.sheet[j]
            if molecule.chain[i] != sheet[0] or resi < sheet[1] or \
               resi > sheet[3]:
                continue
            SecondaryStructures[i] = ord('s')

        for j in range(len(molecule.helix)):
            helix = molecule.helix[j]
            if molecule.chain[i] != helix[0] or resi < helix[1] or \
               resi > helix[3]:
                continue
            SecondaryStructures[i] = ord('h')

    output = vtk.vtkPolyData()

    # for atom type i.e. element
    atom_type = numpy_to_vtk(num_array=elements, deep=True,
                             array_type=vtk.VTK_ID_TYPE)
    atom_type.SetName("atom_type")

    output.GetPointData().AddArray(atom_type)

    # for atom type strings
    atom_types = vtk.vtkStringArray()
    atom_types.SetName("atom_types")
    atom_types.SetNumberOfTuples(num_total_atoms)
    for i in range(num_total_atoms):
        atom_types.SetValue(i, molecule.atom_types[i])

    output.GetPointData().AddArray(atom_types)

    # for residue
    residue = numpy_to_vtk(num_array=molecule.residue_seq, deep=True,
                           array_type=vtk.VTK_ID_TYPE)
    residue.SetName("residue")
    output.GetPointData().AddArray(residue)

    # for chain
    chain = numpy_to_vtk(num_array=molecule.chain, deep=True,
                         array_type=vtk.VTK_UNSIGNED_CHAR)
    chain.SetName("chain")
    output.GetPointData().AddArray(chain)

    # for secondary structures
    s_s = numpy_to_vtk(num_array=SecondaryStructures, deep=True,
                       array_type=vtk.VTK_UNSIGNED_CHAR)
    s_s.SetName("secondary_structures")
    output.GetPointData().AddArray(s_s)

    # for secondary structures begin
    newarr = np.ones(num_total_atoms)
    s_sb = numpy_to_vtk(num_array=newarr, deep=True,
                        array_type=vtk.VTK_UNSIGNED_CHAR)
    s_sb.SetName("secondary_structures_begin")
    output.GetPointData().AddArray(s_sb)

    # for secondary structures end
    newarr = np.ones(num_total_atoms)
    s_se = numpy_to_vtk(num_array=newarr, deep=True,
                        array_type=vtk.VTK_UNSIGNED_CHAR)
    s_se.SetName("secondary_structures_end")
    output.GetPointData().AddArray(s_se)

    # for ishetatm
    ishetatm = numpy_to_vtk(num_array=molecule.is_hetatm, deep=True,
                            array_type=vtk.VTK_UNSIGNED_CHAR)
    ishetatm.SetName("ishetatm")
    output.GetPointData().AddArray(ishetatm)

    # for model
    model = numpy_to_vtk(num_array=molecule.model, deep=True,
                         array_type=vtk.VTK_UNSIGNED_INT)
    model.SetName("model")
    output.GetPointData().AddArray(model)

    # for coloring the heteroatoms
    rgb = vtk.vtkUnsignedCharArray()
    rgb.SetNumberOfComponents(3)
    rgb.Allocate(3 * num_total_atoms)
    rgb.SetName("rgb_colors")

    table = PeriodicTable()
    for i in range(num_total_atoms):
        rgb.InsertNextTuple(table.atom_color(elements[i]))

    output.GetPointData().SetScalars(rgb)

    # for radius of the heteroatoms
    Radii = vtk.vtkFloatArray()
    Radii.SetNumberOfComponents(3)
    Radii.Allocate(3 * num_total_atoms)
    Radii.SetName("radius")

    for i in range(num_total_atoms):
        Radii.InsertNextTuple3(table.atomic_radius(elements[i], 'VDW'),
                               table.atomic_radius(elements[i], 'VDW'),
                               table.atomic_radius(elements[i], 'VDW'))

    output.GetPointData().SetVectors(Radii)

    # setting the coordinates
    points = numpy_to_vtk_points(coords)
    output.SetPoints(points)

    ribbonFilter = vtk.vtkProteinRibbonFilter()
    ribbonFilter.SetInputData(output)
    ribbonFilter.SetCoilWidth(0.2)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(ribbonFilter.GetOutputPort())
    molecule_actor = vtk.vtkActor()
    molecule_actor.SetMapper(mapper)
    make_molecularviz_aesthetic(molecule_actor)
    return molecule_actor
