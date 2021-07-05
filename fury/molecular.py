import vtk
from vtk.util import numpy_support


class Molecule(vtk.vtkMolecule):
    """Your molecule class.

    An object that is used to create molecules and store molecular data (e.g.
    coordinate and bonding data).
    This is a more pythonic version of ``vtkMolecule`` providing simple methods
    for creating molecules. It provides access to essential functionality
    available in ``vtkMolecule``.
    """

    def addAtom(self, atomic_num, x_coord, y_coord, z_coord):
        """Add atomic data to our molecule.

        Parameters
        ----------
        atomic_num : int
            Atomic number of the atom.
        x_coord : float
            x-coordinate of the atom.
        y_coord : float
            y-coordinate of the atom.
        z_coord : float
            z-coordinate of the atom.
        """
        self.AppendAtom(atomic_num, x_coord, y_coord, z_coord)

    def addBond(self, atom1_index, atom2_index, bond_type=1):
        """Add bonding data to our molecule. Establish a bond of type bond_type
        between the atom at atom1_index and the atom at atom2_index.

        Parameters
        ----------
        atom1_index : int
            Index of the first atom.
        atom2_index : int
            Index of the second atom.
        bond_type : int (optional)
            Type of bond (single/double/triple). Default: 1
        """
        self.AppendBond(atom1_index, atom2_index, bond_type)

    def getTotalNumAtoms(self):
        """Returns the total number of atoms in the molecule.
        """
        return self.GetNumberOfAtoms()

    def getTotalNumBonds(self):
        """Returns the total number of bonds in the molecule.
        """
        return self.GetNumberOfBonds()

    def getAtomicNumber(self, atom_index):
        """Get the atomic number of an atom for a specified index.

        Returns the atomic number of the atom present at index atom_index.

        Parameters
        ----------
        atom_index : int
            Index of the atom whose atomic number is to be obtained.
        """
        return self.GetAtomAtomicNumber(atom_index)

    def setAtomicNumber(self, atom_index, atomic_num):
        """Set the atomic number of an atom for a specified index.

        Assign atomic_num as the atomic number of the atom present at
        atom_index.

        Parameters
        ----------
        atom_index : int
            Index of the atom to whom the atomic number is to be assigned.
        atom_num : int
            Atomic number to be assigned to the atom.
        """
        self.SetAtomAtomicNumber(atom_index, atomic_num)

    def getAtomicPosition(self, atom_index):
        """Get the atomic coordinates of an atom for a specified index.

        Returns the atomic coordinates of the atom present at index atom_index.

        Parameters
        ----------
        atom_index : int
            Index of the atom whose atomic coordinates are to be obtained.
        """
        return self.GetAtomPosition(atom_index)

    def setAtomicPosition(self, atom_index, atom_coordinate):
        """Set the atomic coordinates of an atom for a specified index.

        Assign atom_coordinate to the coordinates of the atom present at
        atom_index.

        Parameters
        ----------
        atom_index : int
            Index of the atom to which the coordinates are to be assigned.
        atom_coordinate : int
            Coordinates to be assigned to the atom.
        """
        self.SetAtomPosition(atom_index, atom_coordinate)

    def getBondType(self, bond_index):
        """Get the type of bond for a specified index.

        Returns the type of bond (whether it's a single/double/triple bond)
        present at bond_index.

        Parameters
        ----------
        bond_index : int
            Index of the bond whose type is to be obtained.
        """
        return self.GetBondOrder(bond_index)

    def setBondType(self, bond_index, bond_type):
        """Set the bond type of a bond for a specified index.

        Assign bond_type (whether it's a single/double/triple bond) to the bond
        present at the bond_index.

        Parameters
        ----------
        bond_index : int
            Index of the atom to which the coordinates are to be assigned.
        bond_type : int
            Type of the bond (single/double/triple).
        """
        return self.SetBondOrder(bond_index, bond_type)

    def getAtomicNumberArray(self):
        """Returns an array of atomic numbers corresponding to the atoms
        present in the molecule.
        """
        return numpy_support.vtk_to_numpy(self.GetAtomicNumberArray())

    def getBondTypesArray(self):
        """Returns an array containing the types of the bond (single/double/
        triple) corresponding to the bonds present in the molecule.
        """
        return numpy_support.vtk_to_numpy(self.GetBondOrdersArray())

    def getAtomicPositionArray(self):
        """Returns an array of atomic coordinates corresponding to the atoms
        present in the molecule.
        """
        return numpy_support.vtk_to_numpy(self.GetAtomicPositionArray().
                                          GetData())

    def deepCopy(self, mol):
        """
        Deep copies the atomic information (atoms and bonds) from mol into
        the instance calling the function.

        Parameters
        ----------
        mol : Molecule object
        """
        self.DeepCopyStructure(mol)

    def usevtkBondingAlgorithm(self):
        """
        Uses vtkSimpleBondPerceiver to generate bonding information for a
        molecule.
        vtkSimpleBondPerceiver performs a simple check of all interatomic
        distances and adds a single bond between atoms that are reasonably
        close. If the interatomic distance is less than the sum of the two
        atom's covalent radii plus a tolerance, a single bond is added.

        Notes
        -----
        This algorithm does not consider valences, hybridization, aromaticity,
        or anything other than atomic separations. It will not produce anything
        other than single bonds.
        """
        bonder = vtk.vtkSimpleBondPerceiver()
        bonder.SetInputData(self)
        bonder.SetTolerance(0.1)
        bonder.Update()
        self.deepCopy(bonder.GetOutput())


class MoleculeMapper(vtk.vtkOpenGLMoleculeMapper):
    """Class to create mappers for three types of molecular represenations-
    1. Ball and Stick Representation.
    2. Stick Representation.
    3. Sphere Representation.

    Its member functions are used to create mappers which are then used to
    create actors for the above mentioned representations.
    """

    def setMoleculeData(self, molecule):
        """This member function performs two tasks -
        1. It sends the molecule data to the mapper object.
        2. It checks if adequate bonding data is available and assigns a bool
        to bonds_data_available accordingly.

        Parameters
        ----------
        molecule : Molecule object
            Molecule object whose data is sent to the mapper.
        """
        self.bonds_data_available = False
        self.SetInputData(molecule)
        if molecule.getBondTypesArray().size == molecule.getTotalNumBonds() \
           and molecule.getTotalNumBonds() > 0:
            self.bonds_data_available = True

    def setRenderAtoms(self, choice):
        """Set whether or not to render atoms.

        Parameters
        ----------
        choice : bool
            If choice is True, atoms are rendered.
            If choice is False, atoms are not rendered.
        """
        self.SetRenderAtoms(choice)

    def setRenderBonds(self, choice):
        """Set whether or not to render bonds.

        Parameters
        ----------
        choice : bool
            If choice is True, bonds are rendered.
            If choice is False, bonds are not rendered.
        """
        self.SetRenderBonds(choice)

    def setAtomicRadiusTypeToVDWRadius(self):
        """Set the type of radius used to generate the atoms to Van der Waals
        radius.
        """
        self.SetAtomicRadiusTypeToVDWRadius()

    def setAtomicRadiusTypeToUnitRadius(self):
        """Set the type of radius used to generate the atoms to unit radius.
        """
        self.SetAtomicRadiusTypeToUnitRadius()

    def setAtomicRadiusTypeToCovalentRadius(self):
        """Set the type of radius used to generate the atoms to covalent
        radius.
        """
        self.SetAtomicRadiusTypeToCovalentRadius()

    def setAtomicRadiusScaleFactor(self, scaleFactor):
        """Set the uniform scaling factor applied to the atoms.

        Parameters
        ----------
        scaleFactor : float
            Scaling factor to be applied to the atoms.
        """
        self.SetAtomicRadiusScaleFactor(scaleFactor)

    def setBondColorModeToDiscrete(self, choice):
        """Set the method by which bonds are colored.

        Parameters
        ----------
        choice : bool
            If choice is True, each bond is colored using the same lookup
            table as the atoms at each end, with a sharp color boundary at the
            bond center.
            If choice is False, all bonds will be of same color.
        """
        self.SetBondColorMode(choice)

    def setAtomColorModeToDiscrete(self, choice):
        """Set the method by which atoms are colored.

        Parameters
        ----------
        choice : bool
            If choice is True, each atom is colored using the internal lookup
            table.
            If choice is False, all atoms will be of same color.
        """
        self.SetAtomColorMode(choice)

    def setBondThickness(self, bondThickness):
        """Sets the thickness of the bonds (i.e. thickness of tubes which are
         used to render bonds)
        Parameters
        ----------
        bondThickness: float
            Thickness of the bonds.
        """
        self.SetBondRadius(bondThickness)

    def setMultiTubesForBonds(self, choice):
        """Set whether multiple tubes will be used to represent multiple bonds.

        Parameters
        ----------
        choice : bool
            If choice is True, multiple bonds (double, triple) will be shown by
            using multiple tubes.
            If choice is False, all bonds (single, double, triple) will be
            shown as single bonds (i.e shown using one tube each).
        """
        self.SetUseMultiCylindersForBonds(choice)

    def useMolecularSphereRep(self, colormode):
        """Set Mapper settings to create a molecular sphere representation.

        Parameters
        ----------
        colormode : string
            Set the colormode for coloring the atoms. Two valid color modes -
        * 'discrete': each atom is colored using the internal lookup table.
        * 'single': all atoms of same color.
        """
        self.setRenderAtoms(True)
        self.setRenderBonds(False)
        self.setAtomicRadiusTypeToVDWRadius()
        self.setAtomicRadiusScaleFactor(1.0)
        if colormode == 'discrete':
            self.setAtomColorModeToDiscrete(True)
        elif colormode == 'single':
            self.setAtomColorModeToDiscrete(False)

    def useMolecularBallStickRep(self, colormode, atom_scale_factor,
                                 bond_thickness, multipleBonds):
        """Set Mapper settings to create a molecular ball and stick
        representation.

        Parameters
        ----------
        colormode : string
            Set the colormode for coloring the atoms. Two valid color modes -
            * 'discrete': each atom and bond is colored using the internal
              lookup table.
            * 'single': All atoms are colored with same color(grey) and all
              bonds are colored with same color(dark grey).
            RGB tuple used for coloring the atoms: (150, 150, 150)
            RGB tuple used for coloring the bonds: (50, 50, 50)
        atom_scale_factor : float
            Scaling factor to be applied to the atoms.
        bond_thickness : float
            Used to manipulate the thickness of bonds (i.e. thickness of tubes
            which are used to render bonds)
        multipleBonds : string
            Set whether multiple tubes will be used to represent multiple
            bonds. Two valid choices -
            * 'On': multiple bonds (double, triple) will be shown by using
              multiple tubes.
            * 'Off': all bonds (single, double, triple) will be shown as single
              bonds (i.e shown using one tube each).
        """
        if self.bonds_data_available:
            self.setRenderAtoms(True)
            self.setRenderBonds(True)
            self.setBondThickness(bond_thickness/10)
            self.setAtomicRadiusTypeToVDWRadius()
            self.setAtomicRadiusScaleFactor(atom_scale_factor)
            if multipleBonds == 'On':
                self.setMultiTubesForBonds(True)
            elif multipleBonds == 'Off':
                self.setMultiTubesForBonds(False)
        else:
            print("Inadequate Bonding data")

        if colormode == 'discrete':
            self.setAtomColorModeToDiscrete(True)
            self.setBondColorModeToDiscrete(True)
        elif colormode == 'single':
            self.setAtomColorModeToDiscrete(False)
            self.setBondColorModeToDiscrete(False)

    def useMolecularStickRep(self, colormode, bond_thickness):
        """Set Mapper settings to create a molecular stick representation.

        Parameters
        ----------
        colormode : string
            Set the colormode for coloring the bonds. Two valid color modes -
            * 'discrete': Each bond is colored using the internal lookup table.
            * 'single': All bonds are colored with the same color (dark grey).
            RGB tuple used for coloring the bonds: (50, 50, 50)
            Default is 'discrete'.
        atom_scale_factor : float
            Scaling factor to be applied to the atoms.
        bond_thickness : float
            Used to manipulate the thickness of bonds (i.e. thickness of tubes
            which are used to render bonds)
        """
        if self.bonds_data_available:
            self.setRenderAtoms(True)
            self.setRenderBonds(True)
            self.setBondThickness(bond_thickness/10)
            self.setAtomicRadiusTypeToUnitRadius()
            self.setAtomicRadiusScaleFactor(bond_thickness/10)
        else:
            print("Inadequate Bonding data")

        if colormode == 'discrete':
            self.setAtomColorModeToDiscrete(True)
            self.setBondColorModeToDiscrete(True)
        elif colormode == 'single':
            self.setAtomColorModeToDiscrete(False)
            self.setBondColorModeToDiscrete(False)


class PeriodicTable(vtk.vtkPeriodicTable):
    """ A class to obtain properties of elements (eg: Covalent Radius,
    Van Der Waals Radius, Symbol etc.).

    This is a more pythonic version of ``vtkPeriodicTable`` providing simple
    methods to access atomic properties. It provides access to essential
    functionality available in ``vtkPeriodicTable``. An object of this class
    provides access to atomic information sourced from Blue Obelisk Data
    Repository.
    """

    def getAtomSymbolName(self, atomicNumber):
        """Given an atomic number, returns the symbol associated with the
        element.

        Parameters
        ----------
        atomicNumber : int
            atomic number of the element whose symbol is to be obtained.
        """
        return self.GetSymbol(atomicNumber)

    def getAtomElementName(self, atomicNumber):
        """Given an atomic number, returns the name of the element.

        Parameters
        ----------
        atomicNumber : int
            atomic number of the element whose name is to be obtained.
        """
        return self.GetElementName(atomicNumber)

    def getAtomicNumber(self, elementName):
        """Given a case-insensitive string that contains the symbol or name of
        an element, return the corresponding atomic number.

        Parameters
        ----------
        elementName : string
            Name of the element whose atomic number is to be obtained.
        """
        return self.GetAtomicNumber(elementName)

    def getAtomCovRadius(self, atomicNumber):
        """Given an atomic number, return the covalent radius of the atom.

        Parameters
        ----------
        atomicNumber : int
            atomic number of the element whose covalent radius is to be
            obtained.
        """
        return self.GetCovalentRadius(atomicNumber)

    def getAtomVDWRadius(self, atomicNumber):
        """Given an atomic number, return the Van Der Waals radius of the atom.

        Parameters
        ----------
        atomicNumber : int
            atomic number of the element whose Van Der Waals radius is to be
            obtained.
        """
        return self.GetVDWRadius(atomicNumber)

    def getAtomColor(self, atomicNumber):
        """Given an atomic number, return the RGB tuple associated with that
           element provided by the Blue Obelisk Data Repository.

        Parameters
        ----------
        atomicNumber : int
            atomic number of the element whose RGB tuple is to be obtained.
        """
        return self.GetDefaultRGBTuple(atomicNumber)


def make_molecularviz_aesthetic(molecule_actor):
    """Configure actor propeties to make the molecular visualization
    aesthetically pleasant to see by manipulating the lighting.

    Parameters
    ----------
    molecule_actor : vtkActor
        Actor that represents the molecule to be visualized.
    """
    molecule_actor.GetProperty().SetDiffuse(1)
    molecule_actor.GetProperty().SetSpecular(0.5)
    molecule_actor.GetProperty().SetSpecularPower(90.0)


def molecular_sphere_rep_actor(molecule, colormode='discrete'):
    """Create an actor for sphere molecular representation. It's also referred
    to as CPK model and space-filling model.

    Parameters
    ----------
    molecule : Molecule object
        The molecule to be rendered.
    colormode : string
        Set the colormode for coloring the atoms. Two valid color modes -
        * 'discrete': each atom is colored using the internal lookup table.
        * 'single': all atoms of same color.
        Default is 'discrete'.

    Returns
    -------
    molecule_actor : vtkActor
        Actor created to render the space filling representation of the
        molecule to be visualized.
    """
    msp_mapper = MoleculeMapper()
    msp_mapper.setMoleculeData(molecule)
    msp_mapper.useMolecularSphereRep(colormode)
    molecule_actor = vtk.vtkActor()
    molecule_actor.SetMapper(msp_mapper)
    make_molecularviz_aesthetic(molecule_actor)
    return molecule_actor


def molecular_bstick_rep_actor(molecule, colormode='discrete',
                               atom_scale_factor=0.3, bond_thickness=1,
                               multipleBonds='On'):
    """Create an actor for ball and stick molecular representation.

    Parameters
    ----------
    molecule : Molecule object
        The molecule to be rendered.
    colormode : string
        Set the colormode for coloring the atoms. Two valid color modes -
        * 'discrete': each atom and bond is colored using the internal lookup
          table.
        * 'single': All atoms are colored with same color(grey) and all bonds
          are colored with same color(dark grey).
        RGB tuple used for coloring the atoms: (150, 150, 150)
        RGB tuple used for coloring the bonds: (50, 50, 50)
        Default is 'discrete'.
    atom_scale_factor : float
        Scaling factor to be applied to the atoms.
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
    bs_mapper = MoleculeMapper()
    bs_mapper.setMoleculeData(molecule)
    bs_mapper.useMolecularBallStickRep(atom_scale_factor=atom_scale_factor,
                                       colormode=colormode,
                                       bond_thickness=bond_thickness,
                                       multipleBonds=multipleBonds)
    molecule_actor = vtk.vtkActor()
    molecule_actor.SetMapper(bs_mapper)
    make_molecularviz_aesthetic(molecule_actor)
    return molecule_actor


def molecular_stick_rep_actor(molecule, colormode='discrete',
                              bond_thickness=1):
    """Create an actor for stick molecular representation.

    Parameters
    ----------
    molecule : Molecule object
        The molecule to be rendered.
    colormode : string
        Set the colormode for coloring the bonds. Two valid color modes -
        * 'discrete': Each bond is colored using the internal lookup table.
        * 'single': All bonds are colored with the same color (dark grey)
          RGB tuple used for coloring the bonds: (50, 50, 50)
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
    mst_mapper = MoleculeMapper()
    mst_mapper.setMoleculeData(molecule)
    mst_mapper.useMolecularStickRep(colormode, bond_thickness=bond_thickness)
    molecule_actor = vtk.vtkActor()
    molecule_actor.SetMapper(mst_mapper)
    make_molecularviz_aesthetic(molecule_actor)
    return molecule_actor
