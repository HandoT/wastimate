# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""
import numpy as np
import xml.etree.ElementTree as ET
import itertools
from wastimate.core import cram
import os


# Module-level cache for the decay-chain reference data. This file has thousands of
# nuclide entries and never changes during a run, so parsing it from disk and linearly
# scanning it for every single Package (and every daughter nuclide recursively within
# each Package) was the dominant cost when a model has many user-defined packages.
# Parsing once and indexing by name turns each lookup into an O(1) dict access instead
# of an O(n) scan repeated at every recursion level.
_DECAY_CHAIN_ROOT_CACHE = {}
_DECAY_CHAIN_INDEX_CACHE = {}

def _get_decay_chain_index(file_path):
    if file_path not in _DECAY_CHAIN_INDEX_CACHE:
        if file_path not in _DECAY_CHAIN_ROOT_CACHE:
            _DECAY_CHAIN_ROOT_CACHE[file_path] = ET.parse(file_path).getroot()
        root = _DECAY_CHAIN_ROOT_CACHE[file_path]
        _DECAY_CHAIN_INDEX_CACHE[file_path] = {parent.attrib["name"]: parent for parent in root}
    return _DECAY_CHAIN_INDEX_CACHE[file_path]


# Module-level cache of fully-resolved decay subtrees, keyed by file path and then by
# (decay_chain flag, nuclide name). A nuclide's half-life/branching/daughters/etc. are
# properties of the reference data alone -- they never depend on which Package asked for
# them. So once e.g. "U235" has been walked once for a given decay_chain setting, every
# other Package (of which there may be hundreds, all sharing common nuclides) that also
# contains "U235" can just copy the cached result instead of re-walking the XML tree.
# This is on top of (and separate from) _get_decay_chain_index: that cache only makes a
# single name->element lookup O(1); this cache avoids repeating the whole recursive
# resolution (lookup + recursion into every daughter, granddaughter, etc.) per package.
_NUCLIDE_RESOLUTION_CACHE = {}


class Package:
    """Instances describe radioactive materials modeled in Wastimate. Class
    provides a set of methods to calculate and retrieve material properties.
    """
    
    id_iter = itertools.count()
    
    def __init__(self, inventory, mass=None, volume=None, label=None, mode="atoms", batches=1, radioactive=True, decay_chain=True, secular_equilibrium=None, initialize=None, date=None):
        """Initializes a new Package object with the specified inventory and
        other properties.  Responsible for setting the package's physical
        attributes to preparing the nuclide data for decay calculations.
        
        The constructor performs the following key actions:
            Sets Attributes: It initializes the package's physical properties,
            such as mass and volume, and sets the provided flags like
            radioactive and decay_chain. It also creates empty dictionaries
            (InventoryStates, ActivityStates, etc.) that will later store the
            results of the decay calculations.
        
            Initializes Nuclide Data: If initialize is not provided, the method
            calls the Initialize method to parse the decay data from an XML
            file (decay_chains_endfb71.xml) and build the internal dictionaries
            (HalfLifeDict, BranchingDict, etc.). It also ensures that all
            nuclides in the decay chains of the initial inventory are included
            in the package's inventory, adding them with zero quantity if they
            were not initially specified.

            Handles Inventory Units: It converts the initial inventory values
            to NumPy arrays, sampling from distributions if necessary. If the
            mode is "activity" or "activity_concentration", it converts these
            values to atomic quantities (atoms) using the nuclide's half-life.

            Applies Secular Equilibrium: If secular_equilibrium is specified,
            it calls the decay_sequence method to adjust the inventory to 
            reflect secular equilibrium conditions.

            Performs Decay Correction: If a date is provided, it calculates the
            Bateman matrix and uses the cram16 solver to decay the initial
            inventory to the specified date.

        Parameters
        ----------
        inventory : dict
            A dictionary where the keys are the nuclide labels (e.g., "U235")
            and the values are their initial quantities. The quantity can be a
            single float or an object from scipy.stats that represents a
            distribution.
        mass : float, optional
            The mass of the package. Defaults to None.
        volume : float, optional
            The volume of the package. Defaults to None.
        label : str, optional
            A user-defined label for the package. If not provided, a unique
            label like "Package0" will be generated.
        mode : str, optional
            Specifies the units of the initial inventory. Options are "atoms"
            (default), "activity", or "activity_concentration". The constructor 
            will convert "activity" or "activity_concentration" values to 
            "atoms" for internal calculations.
        batches : int, optional
            The number of batches for which to sample the initial inventory
            distribution. Defaults to 1.
        radioactive : bool, optional
           A flag to indicate whether the package contains radioactive
           material. Defaults to True.
        decay_chain : bool, optional
            A flag to indicate whether to consider the decay chain of the
            nuclides. Defaults to True.
        secular_equilibrium : list, optional
            A list of nuclides for which to assume secular equilibrium.
            This means the daughter nuclide's decay rate is equal to the
            mother's. Can also be "all" or None. Defaults to None.
        initialize : tuple, optional
            A tuple containing pre-initialized nuclide data. This can be used 
            to avoid re-parsing the decay data from the XML file. 
            Defaults to None.
        date : float, optional
            The date (in seconds) to decay the initial inventory to.
            Defaults to None.

        Returns
        -------
        None
        """
        
        # initializes a Package with Mass, Inventory, and optional parameters like radioactivity, decay_chain, and initialize.
        self.Mass = mass
        self.Inventory = inventory
        self.Volume = volume
        
        if label == None:
            self.label = "Package" + str(next(self.id_iter))
        else:
            self.label = label
        
        if batches == 1:
            self.batches = 1
        else:
            self.batches = batches

        self.radioactive         = radioactive
        self.decay_chain         = decay_chain      
        
        self.InventoryStates = {}
        self.ActivityStates  = {}
        self.HeatStates      = {}
        self.HeatSumStates   = {}
        
        self.NuclideIndexDict = {}

        if initialize == None:
            AllNuclides = self.initialize(self.Inventory.keys())
            
            if secular_equilibrium is None:
                self.secular_equilibrium = []
            elif secular_equilibrium == "all":
                self.secular_equilibrium = list(self.Inventory.keys())
            else:
                self.secular_equilibrium = secular_equilibrium
            
            for nuc in AllNuclides:
                if nuc not in self.Inventory:
                    self.Inventory[nuc] = np.array([0.0]*self.batches)

        else:
            (self.HalfLifeDict, self.BranchingDict, self.MotherDaughterDict, self.EnergyDict, self.NuclideDict,
             self.InvNuclideDict, self.InventoryStates, self.ActivityStates, self.HeatStates, self.HeatSumStates) = initialize   
        
        for key, values in self.Inventory.items():
            if isinstance(values, float) or isinstance(values, int):
                self.Inventory[key] = np.array([float(values)]*self.batches)
            elif isinstance(values, np.ndarray):
                pass
            else:
                self.Inventory[key] = values.rvs(self.batches)
                  
        if mode == "activity":
            # Convert activity to atomic content.
            for nuc in self.Inventory.keys():            
                if self.HalfLifeDict[nuc] == np.inf:
                    self.Inventory[nuc] = np.array([0.0]*self.batches)
                else:
                    lmbda = np.log(2) / self.HalfLifeDict[nuc]
                    self.Inventory[nuc] = self.Inventory[nuc] / lmbda
                    
        if mode == "activity_concentration":
            # Convert activity to atomic content.
            for nuc in self.Inventory.keys():            
                if self.HalfLifeDict[nuc] == np.inf:
                    self.Inventory[nuc] = np.array([0.0]*self.batches)
                else:
                    lmbda = np.log(2) / self.HalfLifeDict[nuc]
                    self.Inventory[nuc] = self.Inventory[nuc] / lmbda * self.Mass
        
        if initialize == None:
            self.decay_sequence()
        
        # Correct the decay to a given date if package reference date is known.
        if date is not None:
            self.calculate_bateman()
            N_t = np.array(list(self.Inventory.values()))
            N_t = cram.cram16(self.BatemanMatrix, N_t, date, self.radioactive)
            for idx, nuc in enumerate(self.Inventory.keys()):
                self.Inventory[nuc] = N_t[idx]

            
    def get_mass(self):
        return self.Mass
    
    def get_volume(self):
        return self.Volume

    def get_inventory(self, age):
        return self.InventoryStates[age]

    def get_activity(self, age):
        return self.ActivityStates[age]

    def get_heat(self, age):
        return self.HeatStates[age]
                    
    def decay_sequence(self):
        """Applies the concept of secular equilibrium to the nuclide
        inventory. Secular equilibrium occurs when a parent nuclide has a much
        longer half-life than its daughter nuclide(s), causing the daughter's
        decay rate to essentially equal its production rate from the parent.
        This method calculates the inventory of daughter nuclides based on this
        principle, ensuring their quantity is correct relative to their parent.
        
        Performs the following steps:
            The method iterates through each nuclide specified in the
            secular_equilibrium list.

            For each parent nuclide, it recursively traverses the decay chain
            using the nested decay_sequenceEquilibrium helper function.

            The helper function calculates the progeny_ratio
            (the ratio of a daughter's activity to its mother's activity) and
            accumulates the total ratio for each nuclide in the decay chain.

            Finally, the Inventory of each daughter nuclide is adjusted by
            adding the calculated progeny_ratio multiplied by the parent's
            inventory.

        Returns
        -------
        None
            This method modifies the self.Inventory attribute in-place and does
            not return anything.
        """

        # Be careful with very long-lived radionuclides like Bi-209. Numerical Errors! Change data to stable nuclide.
        def decay_sequence_equilibrium(mother_nuc, EquilibriumDict, ProgenyDict):
            """Recursively finds and calculates the equilibrium ratios for
            a decay chain.
            

            Parameters
            ----------
            other_nuc : str
                The label of the mother nuclide.
            EquilibriumDict : dict
                A dictionary to store the equilibrium ratios for each nuclide.
            ProgenyDict : dict
                A dictionary to store the total progeny ratios.

            Notes
            -----
            Be careful with very long-lived radionuclides like Bi-209. 
            Numerical Errors! If problems surface, one option is to change
            a very long-lived radionuclide to stable nuclide.

            Returns
            -------
            EquilibriumDict : dict
                Updates EquilibriumDict.
            ProgenyDict : dict
                Updates ProgenyDict.
            """
            # Take mother nuclide and duplicate the inventory for its daughter nuclides.
            for didx, daughter_nuc in enumerate(self.MotherDaughterDict[mother_nuc]):
                progeny_ratio = EquilibriumDict[mother_nuc]*self.BranchingDict[mother_nuc][didx]
                EquilibriumDict[daughter_nuc] = progeny_ratio
                if daughter_nuc in ProgenyDict:
                    ProgenyDict[daughter_nuc] += progeny_ratio
                else:
                    ProgenyDict[daughter_nuc] = progeny_ratio
                if daughter_nuc in self.MotherDaughterDict:
                    EquilibriumDict, ProgenyDict = decay_sequence_equilibrium(daughter_nuc, EquilibriumDict, ProgenyDict)
            
            return EquilibriumDict, ProgenyDict

        # Define a function that takes a mother nuclide, and converts the activity to numeric?
        for mother_nuc in self.secular_equilibrium:
            EquilibriumDict, ProgenyDict = decay_sequence_equilibrium(mother_nuc, {mother_nuc:1.0}, {})
            
            for nuc, val in ProgenyDict.items():
                halflife_ratio = self.HalfLifeDict[nuc] / self.HalfLifeDict[mother_nuc]

                if halflife_ratio != 0 and halflife_ratio != np.inf:
                    self.Inventory[nuc] += self.Inventory[mother_nuc] * halflife_ratio * ProgenyDict[nuc]


    def refactor(self, TempNuclideList):
        """Standardizes the data arrays for different nuclide lists.
        Method ensures that all state-related data (e.g., inventory, activity)
        are consistently sized and indexed according to a master list of
        nuclides.

        Parameters
        ----------
        TempNuclideList : list
            The master list of all nuclides to be included in the refactored
            arrays.

        Returns
        -------
        None
            This method modifies the internal state of the Package object
            in-place and does not return anything.
        """

        def refactor_array(batches, TempNuclideList, InvNuclideList, InvArray):
            if TempNuclideList != InvNuclideList:
                # Given a list of nuclides (TempNuclideList), convert inventory (InvArray) into same-sized array (replace nonexisting nuclides with 0).
                Y_combined = np.zeros((len(TempNuclideList), batches))
    
                for val_x, val_y in zip(InvNuclideList, InvArray):
                    if val_x in TempNuclideList:
                        idx = TempNuclideList.index(val_x)
                        Y_combined[idx] = val_y
                
                return Y_combined
            else:
                return InvArray

        for time in self.InventoryStates.keys():
            NucState = self.NuclideDict.keys()
            InvState = self.InventoryStates[time]
            ActState = self.ActivityStates[time]
            
            batches = np.shape(InvState)[1]
            
            self.InventoryStates[time] = refactor_array(batches, TempNuclideList, NucState, InvState)
            self.ActivityStates[time] = refactor_array(batches, TempNuclideList, NucState, ActState)
            
        # Remake the Inventory, NuclideDict and InvNuclideDict
        NewInventory = {}
        NewNuclideDict = {}
        NewInvNuclideDict = {}
        for idx, nuclide in enumerate(TempNuclideList):
            if nuclide in self.Inventory:
                NewInventory[nuclide] = self.Inventory[nuclide]
            else:
                NewInventory[nuclide] = np.zeros(self.batches)

            NewNuclideDict[nuclide] = idx
            NewInvNuclideDict[idx] = nuclide
        
        self.Inventory = NewInventory
        self.NuclideDict = NewNuclideDict
        self.InvNuclideDict = NewInvNuclideDict

    def initialize(self, initial_nuclides):
        """Parses nuclide decay data from an XML file and populates internal
        dictionaries.

        This method is the first step in preparing the nuclide data for decay
        simulation. It identifies all nuclides in the decay chains of the
        initial inventory and gathers their relevant properties.

        Parameters
        ----------
        initial_nuclides : list or iterable
            A list of nuclide labels that make up the initial inventory of the
            package.

        Returns
        -------
        set
            A set containing all unique nuclide labels found in the initial
            inventory and their complete decay chains.
        """

        # initialize dictionaries for nuclide properties
        self.HalfLifeDict = {}
        self.BranchingDict = {}
        self.DecayTypeDict = {}
        self.MotherDaughterDict = {}
        self.EnergyDict = {}
        
        # Construct the absolute path to the data file, and get the cached (parse-once)
        # name -> element index instead of re-parsing and linearly re-scanning the whole
        # file for every package and every nuclide in its decay chain.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, '..', 'data', 'decay_chains_endfb71.xml')
        decay_index = _get_decay_chain_index(file_path)

        # Cache of already-resolved subtrees for this file, keyed by decay_chain mode
        # (True/False resolve differently) and then by nuclide name. A nuclide's own
        # half-life/energy/daughters/branching/decay-type are fixed properties of the
        # reference data, independent of which Package or which of its sibling nuclides
        # triggered the lookup -- so once "U235" (say) has been resolved once for a given
        # decay_chain setting, every other Package containing "U235" reuses that result.
        file_cache = _NUCLIDE_RESOLUTION_CACHE.setdefault(file_path, {})
        mode_cache = file_cache.setdefault(self.decay_chain, {})

        def _resolve(Parent):
            """Resolves (or fetches from cache) the full decay subtree rooted
            at Parent: every nuclide property and every daughter/candidate that
            find_decay_products would have discovered for Parent alone, packaged
            up so it can be merged into any Package's instance dictionaries
            without re-walking the XML tree.

            Returns
            -------
            dict
                {"half_life", "energy", "mother_daughter", "branching",
                 "decay_type", "descendants", "not_found"} -- all dict/list
                values cover Parent plus every nuclide beneath it in its chain.
            """

            if Parent in mode_cache:
                return mode_cache[Parent]

            parent = decay_index.get(Parent)

            if parent is not None:
                parent_name = parent.attrib["name"]

                if "half_life" in parent.attrib:
                    half_life = {parent_name: float(parent.attrib["half_life"])}
                    energy = {parent_name: float(parent.attrib["decay_energy"])}
                    mother_daughter = {}
                    branching = {}
                    decay_type = {}
                    descendants = []

                    if self.decay_chain:
                        CandidateList = []
                        for daughter in parent.iter('decay'):
                            if "target" in daughter.attrib:
                                daughter_name = daughter.attrib["target"]
                            else:
                                daughter_name = parent_name

                            if daughter_name != parent_name:
                                CandidateList.append(daughter_name)

                                if parent_name in mother_daughter:
                                    if daughter_name not in mother_daughter[parent_name]:
                                        mother_daughter[parent_name].append(daughter_name)
                                        branching[parent_name].append(float(daughter.attrib["branching_ratio"]))
                                        decay_type[parent_name].append(daughter.attrib["type"])
                                else:
                                    mother_daughter[parent_name] = [daughter_name]
                                    branching[parent_name] = [float(daughter.attrib["branching_ratio"])]
                                    decay_type[parent_name] = [daughter.attrib["type"]]

                        for candidate in CandidateList:
                            sub = _resolve(candidate)
                            half_life.update(sub["half_life"])
                            energy.update(sub["energy"])
                            mother_daughter.update(sub["mother_daughter"])
                            branching.update(sub["branching"])
                            decay_type.update(sub["decay_type"])
                            descendants.extend(sub["descendants"])
                            descendants.append(candidate)

                        resolved = {"half_life": half_life, "energy": energy,
                                    "mother_daughter": mother_daughter, "branching": branching,
                                    "decay_type": decay_type, "descendants": descendants,
                                    "not_found": False}

                    else:
                        # NOTE: this mirrors the original code's check against
                        # self.MotherDaughterDict, which is never populated when
                        # decay_chain is False -- so the condition is always False
                        # and DecayTypeDict[parent_name] ends up holding only the
                        # *last* decay entry's type rather than a full list. That
                        # quirk is preserved here rather than fixed.
                        never_populated = {}
                        for daughter in parent.iter('decay'):
                            if parent_name in never_populated:
                                decay_type[parent_name].append(daughter.attrib["type"])
                            else:
                                decay_type[parent_name] = [daughter.attrib["type"]]

                        resolved = {"half_life": half_life, "energy": energy,
                                    "mother_daughter": {}, "branching": {},
                                    "decay_type": decay_type, "descendants": [],
                                    "not_found": False}

                    mode_cache[Parent] = resolved
                    return resolved

                # Found in the data, but stable (no half_life attribute).
                resolved = {"half_life": {parent_name: np.inf}, "energy": {parent_name: 0.0},
                            "mother_daughter": {}, "branching": {}, "decay_type": {},
                            "descendants": [], "not_found": False}
                mode_cache[Parent] = resolved
                return resolved

            # Not found in the decay data at all -- treat as stable, self-referencing.
            resolved = {"half_life": {Parent: np.inf}, "energy": {Parent: 0.0},
                        "mother_daughter": {Parent: [Parent]}, "branching": {Parent: [0.0]},
                        "decay_type": {Parent: [None]}, "descendants": [],
                        "not_found": True}
            mode_cache[Parent] = resolved
            return resolved

        def find_decay_products(Parent, History):
            """Merges the resolved (possibly cached) subtree for Parent into
            this Package's instance dictionaries and extends History exactly
            as the original recursive walk would have.

            Parameters
            ----------
            Parent : str
                The name of the parent nuclide to find decay products for.
            History : list
                A list that tracks the nuclides already found.

            Returns
            -------
            History : list
                The updated list of nuclides found in the decay chain.
            """

            resolved = _resolve(Parent)

            self.HalfLifeDict.update(resolved["half_life"])
            self.EnergyDict.update(resolved["energy"])
            for mother, daughters in resolved["mother_daughter"].items():
                self.MotherDaughterDict[mother] = list(daughters)
            for mother, ratios in resolved["branching"].items():
                self.BranchingDict[mother] = list(ratios)
            for mother, types in resolved["decay_type"].items():
                self.DecayTypeDict[mother] = list(types)

            if resolved["not_found"]:
                print(f"Warning: {Parent} was not found in decay data - initializing as stable.")

            History.extend(resolved["descendants"])
            return History

        # Collecting daughter nuclides from initial nuclides
        DaughterNuclides = []
        for nuclide in initial_nuclides:
            DaughterNuclides += find_decay_products(nuclide, list(initial_nuclides))
        
        # Generating a set of simulated nuclides based on the decay chains found (filter out duplicates)
        SimulatedNuclides = set(DaughterNuclides)
        
        return SimulatedNuclides


    def calculate_bateman(self):
        """Builds the Bateman matrix and energy vector for the package's
        inventory.

        The Bateman matrix describes the time evolution of a radioactive
        inventory by modeling the decay of nuclides and the production of their
        daughters. The energy vector stores the decay energy for each nuclide.

        Returns
        -------
        None
            This method modifies the self.BatemanMatrix and self.EnergyVector
            attributes in-place.
        """

        # Initializing matrices to store Bateman matrix and energy vectors
        self.BatemanMatrix = np.zeros((len(self.Inventory.keys()), len(self.Inventory.keys())))
        self.EnergyVector = np.zeros((len(self.Inventory.keys())))
        
        NuclideList = self.Inventory.keys()

        # Iterating through Inventory nuclides to compute Bateman matrix elements
        for nidx, nuc in enumerate(NuclideList):
            LambdaDiag = np.log(2.0) / self.HalfLifeDict[nuc] # Calculate decay constant
            self.NuclideIndexDict[nuc] = nidx # Assign index for the nuclide
            self.BatemanMatrix[nidx, nidx] = -LambdaDiag # Set diagonal element in Bateman matrix
            self.EnergyVector[nidx] = self.EnergyDict[nuc] * 1.60218e-19 # Set energy vector value
        
        # Computing off-diagonal elements in Bateman matrix based on decay relationships
        for mother, daughters in self.MotherDaughterDict.items():
            for didx in range(len(daughters)):
                LambdaBranch = np.log(2.0) / self.HalfLifeDict[mother] # Calculate decay constant for mother nuclide

                # Set off-diagonal elements in Bateman matrix using branching ratios and decay constants
                self.BatemanMatrix[self.NuclideIndexDict[daughters[didx]],
                                   self.NuclideIndexDict[mother]] = (self.BranchingDict[mother][didx]
                                                                     * LambdaBranch)


    def calculate_states(self, stepsize, timesteps, solver="cram16", cont=False):
        """Simulates the radioactive decay of the package's inventory over
        time.

        This method calculates the time evolution of the nuclide inventory,
        activity, and decay heat using a specified Bateman solver.
    
        Parameters
        ----------
        stepsize : float
            The duration of each time step in seconds.
        timesteps : int
            The total number of time steps to simulate.
        solver : str, optional
            The numerical solver to use for the decay calculation ("cram16" or
            "cram48"). Defaults to "cram16".
        cont : bool, optional
            If True, the simulation continues from the last recorded state.
            Defaults to False.
    
        Returns
        -------
        None
            This method updates the internal state dictionaries (e.g.,
            self.InventoryStates) and does not return a value.
        """
        
        # Calculate Bateman matrix initially
        self.calculate_bateman()
        
        cont = len(self.InventoryStates) != 0
        # If not continuing from previous states
        if not cont:
            N_t = np.array(list(self.Inventory.values()))

            # initialize states at time 0
            self.InventoryStates[0] = N_t
            self.ActivityStates[0] = -np.einsum("ii,ik->ik", self.BatemanMatrix, N_t)
            self.HeatStates[0] = np.einsum("i,ij->ij", self.EnergyVector, self.ActivityStates[0])
            self.HeatSumStates[0] = self.HeatStates[0].sum(axis=0)
        else:
            last_timestep = max(list(self.InventoryStates.keys()))
            N_t = np.array(self.InventoryStates[last_timestep])
        
        # Build the CRAM operator once -- BatemanMatrix and stepsize don't change
        # across this loop, so the (previously per-call) sparse LU factorizations
        # only need to happen once, not once per timestep.
        if solver == "cram48":
            cram_op = cram.build_cram48_operator(self.BatemanMatrix, stepsize, self.radioactive)
        else:
            cram_op = cram.build_cram16_operator(self.BatemanMatrix, stepsize, self.radioactive)

        # Perform iterations for given timesteps
        for step in range(timesteps):
            if solver == "cram48":
                N_t = cram.apply_cram48_operator(cram_op, N_t)
            else:
                N_t = cram.apply_cram16_operator(cram_op, N_t)
                
            ActivityVector = -np.einsum("ii,ik->ik", self.BatemanMatrix, N_t)
            HeatVector = np.einsum("i,ij->ij", self.EnergyVector, ActivityVector)
            
            # Update states based on the step and continuation status
            if not cont:
                self.InventoryStates[(step+1)*stepsize] = N_t
                self.ActivityStates[(step+1)*stepsize] = ActivityVector
                self.HeatStates[(step+1)*stepsize] = HeatVector
                self.HeatSumStates[(step+1)*stepsize] = HeatVector.sum(axis=0)
            else:
                self.InventoryStates[last_timestep+(step+1)*stepsize] = N_t
                self.ActivityStates[last_timestep+(step+1)*stepsize] = ActivityVector
                self.HeatStates[last_timestep+(step+1)*stepsize] = HeatVector
                self.HeatSumStates[last_timestep+(step+1)*stepsize] = HeatVector.sum(axis=0)