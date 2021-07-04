import numpy as np
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
        return numpy_support.vtk_to_numpy(self.GetAtomicPositionArray().\
                                          GetData())

    def deepCopy(self, mol):
        """
        Deep copies the atomic information (atoms and bonds) from mol into
        the instance calling the function.

        Parameters
        ----------
        mol : an instance of class Molecule.
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
