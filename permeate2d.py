from json import load, dump
from glob import glob
from networkx.convert_matrix import from_numpy_matrix
from networkx.algorithms.cycles import minimum_cycle_basis
import numpy as np
from scipy.optimize import leastsq
from os.path import basename
from tqdm import tqdm

from ase import Atoms, io
from ase.visualize import view



class Permeate2D:
    def __init__(self, geometry):
        '''
        Load geometry from file or as ase.Atoms.

        Paramters
        ---------
        geometry : str or ase.Atoms
            Geometry of the 2D flake.

        '''

        if isinstance(geometry, Atoms):
            self.atoms = geometry

        elif isinstance(geometry, str):
            self.atoms = io.read(geometry)

        else:
            raise ValueError('Specify either a valid filename or an ase.Atoms object.')
       
        self.rings = []
        

    def add_rings(self, ring_information):
        '''
        Manually add rings for permeation.

        Parameters
        ----------
        ring_information : list
            Manual specification of ring system. Either a list with indexes to be added or a list of lists with the indexes to add multiple rings.
            
        '''
        
        if isinstance(ring_information[0], int):
            self._add_single_ring(ring_information)
            
        elif isinstance(ring_information[0], list):
            for ring in ring_information:
                self._add_single_ring(ring_information)
                

    def _add_single_ring(self, ring_information):
        '''
        Functionality for addition of information for a single ring.
        
        Parameters
        ----------
        ring_information : list
            Manual specification of one ring system.
            
        '''
        
        added_ring = [x for x in ring_information]
        self.rings = self.rings + added_ring

    
    def find_rings(self, n, visualize=True, vis_species='S', **kwargs):
        '''
        Find rings of desired size in geometry file. Stores result in self.rings.

        Parameters
        ----------
        
        n : int
            Number of edges per ring (n in 'n-membered rings').
        visualize : bool
            If true, window with structure and detected rings will open (default: True).
        vis_species : str
            Change the species as which the ring atoms are shown (default: S).
        '''
       
        # Get atomic indexes of all rings with the desired size (this includes double-counting)
        atomic_indexes = self.__network_analysis(self.atoms.get_positions(), n=n)
        
        # Sort all sub-lists with atomic indexes
        [x.sort() for x in atomic_indexes]
        
        # Declare rings, which contains all unique rings
        rings = []

        for x in atomic_indexes:
            if x not in rings:
                rings.append(x)
        
        # Finally, set self.rings
        self.rings = self.rings + rings
        
        # The following lines are for quality control, so the identified rings are visualized
        if visualize:
            self.view(vis_species=vis_species, **kwargs)


    def reset_rings(self):
        '''
        Reset ring information
        '''
        
        self.rings = []


    def load_rings(self, filename):
        '''
        Load atom indexes that form rings of a certain size (5-, 6-, or 7-membered rings, for example).

        Parameters
        ----------
        filename : str
            File that contains ring information.
        '''
        
        # Load from file
        with open(filename) as f:
            ring_data = load(f)
        
        # Load atomic indexes
        atomic_indexes = ring_data['indexes']
        self.rings = atomic_indexes
        
        # Parent geometry
        self.atoms = Atoms.fromdict(io.jsonio.decode(ring_data['parent_geometry']))


    def save_rings(self, filename):
        '''
        Write ring data to a file for later access.

        Parameters
        ----------
        filename : str
            Name of file to store the found rings
        '''
        
        # Initialise dict
        ring_data = {}

        # Save parent structure
        ring_data['parent_geometry'] = io.jsonio.encode(self.atoms.todict())

        # Add atomic indexes to ring_order
        ring_data['indexes'] = self.rings
        
        # Write the JSON file with the given file name
        with open(filename, 'w') as f:
            dump(ring_data, f)    


    def view(self, vis_species='S', **kwargs):
        '''
        

        Parameters
        ----------
        vis_species : string, optional
            Species as which the found ring-forming atoms are displayed. The default is 'S'.

        Returns
        -------
        None.

        '''
                
        atoms_vis = self.atoms.copy()
    
        for i in range(len(self.rings)):
            for j in range(len(self.rings[i])):
                atoms_vis[self.rings[i][j]].symbol = vis_species

        view(atoms_vis, **kwargs)
        

    def generate(self, max_distance, increment, species='H'):
        '''
        Core functionality. Takes XYZ geometry files and atomic indexes of rings, fits plane and adds atoms in given range with given increment.

        Parameters
        ----------
        max_distance : float
            Maximum distance to either side of the ring.
        increment : float
            Increment in which the atoms are put.
        species : str
            Atomic symbol of species which should be added.

        Returns
        -------
        traj : list
            List with ase.Atoms objects.
        '''

        traj = []
        # Loop over all rings
        for idx, ring in enumerate(self.rings):
            tmp = []
                            
            # Find the respective plane and centroid
            plane_parameters, centroid = self._find_plane(self.atoms, ring, verbose=False)
            
            # Get list with all H_coords
            H_coords = self._create_Hs(plane_parameters, centroid, max_distance, increment)
            
            # Loop over all distances for H atoms
            for h, H in enumerate(H_coords):
                
                # Load the geometry file from the working dir
                atoms = self.atoms.copy()
                
                # Create ASE Atoms object with the additional H atom
                H_atom = Atoms(symbols=[species], positions=[H])
                
                # Add H Atoms object to Atoms object loaded from geometry file
                atoms.extend(H_atom)
                
                tmp.append(atoms)
                
            traj.append(tmp)

        return traj

            
    def _create_Hs(self, plane_parameters, centroid, max_distance, increment):
        '''
        Generate a list of coordinates of additional atoms that cross a selected polygon through its centroid normal axis.

        Parameters
        ----------
        plane_parameters : list
            Parameters defining the plane in the form [A, B, C, D] for a plane with Ax + By + Cz + D = 0.
        centroid : list
            Coordinates of the centroid (i.e. center of the polygon).
        Range : float
            Extremes of the distances the additional atoms should be put.
        increment : float
            In which distances the additional atoms should be placed.

        Returns
        -------
        Hs : list
            List of coordinates of the additional atoms.

        '''
        
        # Declare Hs, the list which contains all the coordinates of additional atoms
        Hs = []
        
        # Scale normal vector of the plane to lenght 1
        normal = plane_parameters[0:3]/np.sqrt(np.sum(plane_parameters[0:3]**2))
        
        # Loop over all positions specified by Range and increment to add these positions to Hs
        for i in np.arange(-max_distance, max_distance+increment, increment):
            Hs.append(centroid+normal*i)
            
        return Hs


    def _find_plane(self, atoms, atomic_indexes, verbose=False):
        '''
        Fit plane to given list of atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            ase.Atoms object
        atomic_indexes : list
            Atom indexes of atoms being part of the ring. Pythonic counting starting with 0!
        verbose : bool, optional
            Specify whether information of fitted plane should be printed. The default is False.

        Returns
        -------
        plane_parameters : list
            Parameters [A, B, C, D] defining the fitted plane with Ax + By + Cz +D = 0.
        centroid : list
            XYZ coordinates of the centroid.

        '''
        
        # Retrieve all relevant atomic positions
        pos = [atoms.get_positions()[i] for i in atomic_indexes]
        
        # Sort those positions to each dimension and summarize in numpy array
        x = [x[0] for x in pos]
        y = [y[1] for y in pos]
        z = [z[2] for z in pos]
        XYZ = np.array([x, y, z])    

        # Default approach for plane being a plane in xy-plane with z=0
        # p0 = [A, B, C, D] for planes with Ax + By + Cz + D = 0
        p0 = [0, 0, 1, 0]

        # Get optimised plane parameters
        plane_parameters = leastsq(self.__residuals_plane, p0, args=(XYZ))[0]
        
        # Some output for better understanding what the code does
        if verbose:
            print("Solution: ", plane_parameters)
            print("Old Error: ", (self.__residuals_plane(p0, XYZ)**2).sum())
            print("New Error: ", (self.__residuals_plane(plane_parameters, XYZ)**2).sum())
            
        # Calculate centroid position for that plane
        centroid = self.__centroid(pos)
        
        return plane_parameters, centroid

        
    def __f_to_min(self, coordinates, plane_parameters):
        '''
        Parameters
        ----------
        coordinates : list
            List of points the distance to the plane should be calcuated for.
        plane_parameters : list
            Parameters A, B, C, D, that specify the plane of the format Ax + By + Cz + D = 0.

        Returns
        -------
        distance : list
            Distances of each point to the plane.

        '''

        # Extract A, B, and C from plane parameters
        plane_xyz = plane_parameters[0:3]
        
        coordinates = np.array(coordinates)
        
        
        # Calculate counter
        distance = (plane_xyz*coordinates.T).sum(axis=1) + plane_parameters[3] #p[3] being the right part in the plane equation (D)
        
        # Divide counter through denominator to get distance
        distance = distance / np.linalg.norm(plane_xyz)

        return distance


    def __residuals_plane(self, params, coords):
        return self.__f_to_min(coords, params)


    def __centroid(self, points):
        '''
        Calculate centroid from list of points.

        Parameters
        ----------
        points : list
            List of points the centroid should be calculated for.

        Returns
        -------
        centroid : list
            Coordinates of the centroid.

        '''
        
        # Simply calculate the centroid as the mean of each x,y,z-coordinate of the points
        centroid = np.mean(points, axis=0)
        return centroid


    def __network_analysis(self, positions, n):
        '''
        Actual network analysis based on graph theory.

        Parameters
        ----------
        positions : list
            List of atomic positions.
        n : int
            Ring size the algorithm should look for.

        Returns
        -------
        atomic_indexes : list
            List of atomic indexes that make a ring or that make rings of considered size.

        '''
        
        # Initialise list which should later contain the atomic indexes of rings
        atomic_indexes = []
        
        # This is a parameter for finding next neighbors. Should not be too tight
        next_neighbor_cutoff = 5
        
        # Loop over all atoms
        for i in tqdm(range(len(positions))):
            
            # Initialise list of atoms that is inside the cutoff region
            selected_neighbors = []
            
            # Loop over all atoms
            for j in range(len(positions)):
                
                # If the distance of the first considered atom i to the second considered atom j is below 5 Angstrom, add j's atom index.
                # At the moment of writing, only rings up to 7-membered rings are of interest. Therefore, a cutoff of 5 Angstrom seems
                # to be enough. Should be adjusted if larger rings are investigated. However, this choice is the most time-relevant,
                # since the networkx analysis algorithm has to take all edges (=all bonds) into account.
                if np.linalg.norm(positions[j]-positions[i]) < next_neighbor_cutoff:
                    selected_neighbors.append(j)
            
            # selected_positions being a list of atomic positions of all neighbors of i (neighbors = in the cutoff region)
            # and append atom i itself
            selected_positions = [positions[j] for j in range(len(positions)) if j in selected_neighbors]
            selected_positions.append(positions[i])
            
            # Initialise the adjacency matrix with zeros. The adjacency matrix has a 0 if atom m and n are no neighbors, and a 1
            # if atoms i and j are neighbors. There is a cutoff radius of 1.5 Angstroms. All atoms within the region are considered
            # direct neighbors, all atoms outside that are not considered neighbors. If problems occur, this could possible be adjusted
            # to about 1.7 Angstrom. However, for graphenoid systems, 1.5 Angstroms should be save to neither miss neighbors nor to
            # find false positives.
            # Note: After experience that 1.5 Angstrom cutoff was not enough, cutoff was set to 1.7 with advice to thoroughly check result.
            adjacency = np.zeros((len(selected_positions), len(selected_positions)))
            for j in range(len(selected_positions)):
                for k in range(len(selected_positions)):
                    if j != k:
                        if np.linalg.norm(selected_positions[k]-selected_positions[j]) < 1.7:
                            adjacency[j][k] = 1
            
            # Create Graph from adjacency matrix
            G = from_numpy_matrix(adjacency)
            
            # Find all minimum cycles for each atom (i.e. find ring sizes of all rings and return their indexes)
            # ATTENTION: These are the indexes of atoms in the selected region, not the entire geometry!
            cycles = minimum_cycle_basis(G)
            
            # Filter for rings with desired size
            cycles = [x for x in cycles if len(x) == n]
    
            # Often, no ring of desired size is found. Then, this step should be skipped.
            # If applicable, then the indexes of atoms in the selected region should be transformed to indexes of the entire geometry
            if len(cycles) != 0:
                for j in range(len(cycles)):
                    cycles[j] = [selected_neighbors[x] for x in cycles[j]]
            
            # Finally, add the found cycles for atom i to the global list of atomic indexes making up rings with desired size
            for j in range(len(cycles)):
                atomic_indexes.append(cycles[j])
                
                
        atomic_indexes = set(tuple(i) for i in atomic_indexes)
        atomic_indexes = [list(i) for i in atomic_indexes]

        return atomic_indexes
        
    
class Analyzer:
    def __init__(self, calcdir='.'):
    
        files = glob(calcdir+'/'+'*/'+'*output')
        #files= glob('*output')
        
        results = [['nSnapshot', 'nRing', 'distance / Ang', 'total Energy / eV']]
        
        for f in files:
            f_basename = basename(f).split('.')[0] 
            
            nSnapshot = int(f_basename.split('_')[0][1:])
            nRing = int(f_basename.split('_')[1])
            distance = float(f_basename.split('_')[2]) / 100

            with open(f) as fi:
                lines = fi.readlines()
            
            totalEnergy = [x for x in lines if 'Total energy' in x]
            totalEnergy = float(list(filter(None, totalEnergy[-1].split()))[2])
            
            results.append([nSnapshot, nRing, distance, totalEnergy])
    
            
        self.results = results
        self.write()
    
        
    def write(self):    
        with open('results.json', 'w') as f:
            dump(self.results, f)
        
        with open('results.txt', 'w') as f:
            for i in range(len(self.results)):
                f.write('%s %s %s %s\n' % (self.results[i][0], self.results[i][1], self.results[i][2], self.results[i][3]))
