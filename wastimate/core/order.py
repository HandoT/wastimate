# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""

import os
import copy
import numpy as np
import xml.etree.ElementTree as ET
import itertools
from wastimate.core import combinedpackage as cge

# Module-level cache for the decay-chain reference data -- see package.py for the full
# explanation. This file has thousands of entries and never changes during a run, so
# parsing it and linearly scanning it on every single call (recursively, for every
# nuclide in every chain) was extremely wasteful. Parse once, index by name, O(1) lookup.
_DECAY_CHAIN_ROOT_CACHE = {}
_DECAY_CHAIN_INDEX_CACHE = {}

def _get_decay_chain_index(file_path):
    if file_path not in _DECAY_CHAIN_INDEX_CACHE:
        if file_path not in _DECAY_CHAIN_ROOT_CACHE:
            _DECAY_CHAIN_ROOT_CACHE[file_path] = ET.parse(file_path).getroot()
        root = _DECAY_CHAIN_ROOT_CACHE[file_path]
        _DECAY_CHAIN_INDEX_CACHE[file_path] = {parent.attrib["name"]: parent for parent in root}
    return _DECAY_CHAIN_INDEX_CACHE[file_path]


# Module-level cache of a nuclide's fully-resolved, flattened descendant chain, keyed by
# file path and then by nuclide name. Which daughters/granddaughters/etc. a nuclide has
# is a fixed property of the reference data, not of which Order or which sibling nuclide
# triggered the lookup -- so once "U235" (say) has been walked once, every other Order
# that also references "U235" reuses the flattened result instead of re-recursing through
# the tree. This sits on top of (and is separate from) _get_decay_chain_index, which only
# makes a single name->element lookup O(1); this cache avoids repeating the recursion
# into every daughter, granddaughter, etc. altogether.
_DECAY_DESCENDANTS_CACHE = {}

class Order:
    
    id_iter = itertools.count()
    
    def __init__(self, homenode, ordernodes, label=None, magnitude=1, mode="transfer", remaindernode=None, rate=0, direction="old",
                 mixratio=0.5, criteria=None, extra_criteria=None, instruct=None, crumbs=False, centile=80):
        """Initializes an Order object to manage package transfers between
        nodes.
     
        Parameters
        ----------
        homenode : object
            The destination Node for packages.
        ordernodes : list of Node objects
            A list of source Nodes from which packages will be moved.
        label : str, optional
            A user-defined label for the order. Defaults to a generated label.
        magnitude : int, float, or scipy.stats object, optional
            The number of packages to move. Defaults to 1.
        mode : str, optional
            The type of transfer (e.g., "transfer"). Defaults to "transfer".
        remaindernode : object, optional
            A Node for packages not meeting the criteria. Defaults to None.
        rate : float, optional
            The time interval between transfers. Defaults to 0.
        direction : str, optional
            The order in which packages are selected. Defaults to "old".
        mixratio : float, optional
            The ratio of new to old packages for sorting. Defaults to 0.5.
        criteria : dict or list of dict, optional
            Criteria for selecting packages to be moved. Defaults to a 
            pre-defined criteria.
        extra_criteria : object, optional
            Additional filtering criteria. Defaults to None.
        instruct : dict, optional
            Instructions for package modification. Defaults to None.
        crumbs : bool, optional
            Allows for transferring a smaller number of packages if the
            magnitude is not met. Defaults to False.
        centile : int, optional
            A percentile value for selection. Defaults to 80.
        """
        
        if label == None:
            self.label = "Order" + str(next(self.id_iter))
        else:
            self.label = label
        
        # Create the same way as a link. Instead of awaynode,  define a list of AwayNodes, search for packages in that list that satisfy the criteria.
        self.HomeNode = homenode
        self.OrderNodes = ordernodes
        
        self.Magnitude = magnitude # Number of packages to move OR a criteria for movement.
        self.MagnitudeSample = self.Magnitude # Storing a sample of the magnitude
        self.Rate = rate # Rate of movement
        self.Mode = mode
        self.RemainderNode = remaindernode
        self.UntilDelivery = rate # Store rate for tracking delivery
        
        self.Instruct = instruct # Modification instructions for movement
        
        self.Direction = direction # Order for movement (default: "old")
        self.MixRatio  = mixratio # Ratio of new packages to old packages (default: 50%)
        
        self.Crumbs    = crumbs
        self.extra_criteria = extra_criteria
        
        self.Centile = centile
        
        # Default criteria for movement based on inventory level of a specific nuclide
        Default_Criteria = {"region"   : "package",
                            "variable" : "inventory",
                            #"criteria" : 1, # Default inventory criteria
                            "principle": "min"} # Default nuclide for criteria
        
        #{"sort_nuclides":["Sr90", "Co60"],
        # "sort_factors":[1, 1],
        # "sort_node":Node3}
        
        #Instruct={"package_out":3,
        #          "package_mass":10}
        
        # Handling provided or default criteria for movement
        if criteria != None:
            self.Criteria = criteria
            for Crit in criteria:
                # Update with default criteria for missing keys
                for key, value in Default_Criteria.items():
                    if key not in Crit:
                        Crit[key] = value
        else:
            self.Criteria = None # Use provided criteria or default to None if not provide
            

    def transfer(self, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, centile, transferdate):
        """Performs a transfer of packages based on the order's defined mode.
    
        This method is a comprehensive transfer function that handles various
        modes of operation, including simple transfer, combining, separating,
        and sorting packages.
    
        Parameters
        ----------
        stepsize : float
            The time duration of each simulation step.
        timesteps : int
            The total number of time steps in the simulation.
        UniquePackageDict : dict
            A dictionary of unique package identifiers.
        InvUniquePackageDict : dict
            An inverse dictionary for unique packages.
        centile : int
            A percentile value used in the package suitability criteria.
        transferdate : float
            The date of the transfer.
    
        Returns
        -------
        tuple
            A tuple containing the updated UniquePackageDict and
            InvUniquePackageDict.
        """
            
        # There are multiple modes: Regular Order of Package/s, Order of a part of a package.
        # Make it possible to combine the ordered packages into one new package.
        # Make it possible to modify the package (e.g. volume/mass)

        # Sample from the Magnitude distribution, convert the distribution into a temp int value.
        if isinstance(self.Magnitude, int) or isinstance(self.Magnitude, float):
            self.MagnitudeSample = int(round(self.Magnitude))
        else:
            self.MagnitudeSample = int(self.Magnitude.rvs(1))
            
        # Sample from the package_out distribution, convert the distribution into a temp int value.
        if self.Mode == "batch":
            self.NumOfPacksSample = 1
        elif self.Instruct is None:
            pass
        elif isinstance(self.Instruct["package_out"], int) or isinstance(self.Instruct["package_out"], float):
            self.NumOfPacksSample = int(self.Instruct["package_out"])
        else:
            self.NumOfPacksSample = int(self.Instruct["package_out"].rvs(1))
        
        # Go through the AwayNodes and find one that has fitting characteristics.
        # This means that it must Evaluate each Node, and rank them.
        # Ranking requires a criteria. Criteria is passed using the self.Criteria.
        # For Each criteria, there is True, False, or a number, thus returning a dict of node -> [True, True, False] e.g.
        # Pick either the list with the total highest sum, or highest sum and > the number of entires, or first in priority and > the number of entires.
        # If use ratio between criteria and value to assess the number.
        SuitableNodes = []
        for AwayNode in self.OrderNodes:
            PackageSuitability = self.check_criteria(AwayNode, centile)
            if all(PackageSuitability) and len(PackageSuitability) > 0:
                SuitableNodes.append(AwayNode)
         
        if len(SuitableNodes) != 0:
            indeces = []
            for suit_node in SuitableNodes:
                index = self.OrderNodes.index(suit_node)
                indeces.append(index)
            indeces.sort()
            
            TransferNodes = []
            for idx in indeces:
                TransferNodes.append(self.OrderNodes[idx])
                
                if len(self.OrderNodes[idx].PackageList) > self.MagnitudeSample or len(indeces) == 0:
                    break
            # last and first options while constructing the AvailablePackages
            if self.Direction == "new":
                AvailablePackages = [obj for instance in TransferNodes for obj in instance.PackageList]
            elif self.Direction == "old":
                AvailablePackages = [obj for instance in TransferNodes for obj in list(reversed(instance.PackageList))]
            elif self.Direction == "mix":
                NewPackages = [obj for instance in TransferNodes for obj in instance.PackageList]
                OldPackages = [obj for instance in TransferNodes for obj in list(reversed(instance.PackageList))]                
                PackCount = len(OldPackages)
                NewPackCount = int(round(PackCount * (1 - self.MixRatio)))
                OldPackCount = PackCount - NewPackCount
                AvailablePackages = OldPackages[OldPackCount:] + NewPackages[NewPackCount:]

            NodePackageContributions = []
            total = 0
            for num in [len(nod.PackageList) for nod in TransferNodes]:
                total += num
                NodePackageContributions.append(total)

            if self.Mode == "transfer" or self.Mode == "combine"  or self.Mode == "batch" or self.Mode == "modify": 
                if len(AvailablePackages) >= self.MagnitudeSample: 
                    TempMagnitudeSample = self.MagnitudeSample
                elif len(AvailablePackages) < self.MagnitudeSample and self.Crumbs:
                    TempMagnitudeSample = len(AvailablePackages)
                else:
                    TempMagnitudeSample = 0
        
                # Transfer the packages between nodes.
                if self.Mode == "combine":
                    UniquePackageDict, InvUniquePackageDict = self.combine_packages(TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, transferdate)
                else:
                    mag_idx = 0
                    for mag in range(TempMagnitudeSample):
                        # All the filters need to go here.
                        # Select the removable package state.
                        selected_package_state = AvailablePackages[0] # Oldest packaging.
                            
                        # Remove state from old node.
                        if mag < NodePackageContributions[mag_idx]:
                            TransferNodes[mag_idx].delete_package(selected_package_state)
                        else:
                            mag_idx += 1
                            TransferNodes[mag_idx].delete_package(selected_package_state)

                        AvailablePackages.remove(selected_package_state)
                        # Add state to new node.
                        self.HomeNode.add_package(selected_package_state.copy())

            if self.Mode == "separate":       
                # Transfer the packages between nodes.                            
                UniquePackageDict, InvUniquePackageDict = self.separate(TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, transferdate)

            if self.Mode == "sort":
                # Instruct must have an additional parameters {"remainder":Node, "separated_nuclides":[], "Separation_factor":[]}.
                # Transfer the packages between nodes.                             
                UniquePackageDict, InvUniquePackageDict = self.sort(TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, transferdate)

        return UniquePackageDict, InvUniquePackageDict


    def transfer_all(self, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, centile, transferdate):
        """Performs a simple transfer of packages from source nodes to the home
        node.

        This method samples a magnitude, identifies all suitable packages across
        the source nodes based on the order's criteria, and then transfers the
        oldest of these packages to the home node.
    
        Parameters
        ----------
        stepsize : float
            The time duration of each simulation step.
        timesteps : int
            The total number of time steps in the simulation.
        UniquePackageDict : dict
            A dictionary of unique package identifiers.
        InvUniquePackageDict : dict
            An inverse dictionary for unique packages.
        centile : int
            A percentile value used in the package suitability criteria.
        transferdate : float
            The date of the transfer.
    
        Returns
        -------
        tuple
            A tuple containing the updated UniquePackageDict and
            InvUniquePackageDict.
        """
        
        # Sample from the Magnitude distribution, convert the distribution into a temp int value.
        if isinstance(self.Magnitude, int) or isinstance(self.Magnitude, float):
            self.MagnitudeSample = int(round(self.Magnitude))
        else:
            self.MagnitudeSample = int(self.Magnitude.rvs(1))
        
        AvailablePackages = []
        TempNodePackageContributions = []
        TransferNodes = []
        for AwayNode in self.OrderNodes:
            PackageSuitability = self.check_criteria(AwayNode, centile)
            if any(PackageSuitability):
                TransferNodes.append(AwayNode)
                SuitablePackages   = [element for boolean, element in zip(PackageSuitability, AwayNode.PackageList) if boolean]
                AvailablePackages += SuitablePackages
                TempNodePackageContributions.append(len(SuitablePackages))
                
        NodePackageContributions = []
        total = 0
        for num in TempNodePackageContributions:
            total += num
            NodePackageContributions.append(total)
        
        # Reduce the size of AvailablePackages to MagnitudeSample.
        if len(AvailablePackages) >= self.MagnitudeSample:
                AvailablePackages = AvailablePackages[:self.MagnitudeSample]
        elif len(AvailablePackages) < self.MagnitudeSample and self.Crumbs and len(AvailablePackages) > 0:
                AvailablePackages = AvailablePackages
        elif (len(AvailablePackages) < self.MagnitudeSample and not self.Crumbs) or len(AvailablePackages) == 0:
            return UniquePackageDict, InvUniquePackageDict

        if self.Mode == "transfer":
            mag_idx = 0
            for mag in range(len(AvailablePackages)):
                # All the filters need to go here.
                # Select the removable package state.
                selected_package_state = AvailablePackages[0] # Oldest packaging.
                    
                # Remove state from old node.
                if mag < NodePackageContributions[mag_idx]:
                    TransferNodes[mag_idx].delete_package(selected_package_state)
                else:
                    mag_idx += 1
                    TransferNodes[mag_idx].delete_package(selected_package_state)

                AvailablePackages.remove(selected_package_state)
                # Add state to new node.
                self.HomeNode.add_package(selected_package_state.copy())

        return UniquePackageDict, InvUniquePackageDict


    def sort(self, TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, CreationDate):
        """Sorts a package's contents based on a pre-defined factor,
        distributing the parts to different destination nodes.
    
        This method takes a collection of packages, removes them from their
        source nodes, and then splits them into two new packages based on a
        user-defined sort factor. The resulting packages are sent to the home
        node and a designated sort node.
    
        Parameters
        ----------
        TransferNodes : list of Node
            A list of the source Node objects.
        AvailablePackages : list of tuple
            A list of tuples, each containing the age and Package object.
        NodePackageContributions : list of int
            A list tracking the cumulative number of packages from each 
            transfer node.
        stepsize : float
            The duration of a single simulation time step.
        timesteps : int
            The total number of time steps in the simulation.
        UniquePackageDict : dict
            A dictionary mapping unique package IDs to their objects.
        InvUniquePackageDict : dict
            An inverse dictionary mapping Package objects to their IDs.
        CreationDate : float
            The current simulation time, which becomes the creation date for
            new packages.
    
        Returns
        -------
        tuple
            A tuple containing the updated UniquePackageDict and 
            InvUniquePackageDict.
        """
        
        if len(AvailablePackages) >= self.MagnitudeSample or self.Crumbs:
            if len(AvailablePackages) < self.MagnitudeSample and self.Crumbs:
                InputNumber = len(AvailablePackages)
            else:
                InputNumber = self.MagnitudeSample
            
            CombinationStates = []
            # First create a list of states to be combined.
            if InputNumber != 0:
                mag_idx = 0
                combined_label = ""
                for mag in range(InputNumber):
                    time, selected_package = AvailablePackages[0]
                    CombinationStates.append([time, selected_package])
                    combined_label += selected_package.label + "_"
                    
                    # Remove state from old node.
                    if mag < NodePackageContributions[mag_idx]:
                        TransferNodes[mag_idx].delete_package([time, selected_package])
                    else:
                        mag_idx += 1
                        TransferNodes[mag_idx].delete_package([time, selected_package])

                    AvailablePackages.remove([time, selected_package])
        
        combined_label = combined_label[:-1]
        
        # Refactor coefficients.
        Sep_Coef = self.SortFactors
        Rem_Coef = 1 - Sep_Coef

        # 1.5 Create proper package object without reading the nuclide data.
        separated_package = cge.CombinedPackage(PackageStates=list(zip(CombinationStates, [Sep_Coef]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict, HalfLifeDict=selected_package.HalfLifeDict, DecayTypeDict=selected_package.DecayTypeDict, CreationDate=CreationDate)
        remaining_package = cge.CombinedPackage(PackageStates=list(zip(CombinationStates, [Rem_Coef]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict, HalfLifeDict=selected_package.HalfLifeDict, DecayTypeDict=selected_package.DecayTypeDict, CreationDate=CreationDate)
        
        separated_package.label = f"S{separated_package.id_iter}_"+combined_label
        remaining_package.label = f"R{remaining_package.id_iter}_"+combined_label
        
        if separated_package not in UniquePackageDict.values():
            if len(UniquePackageDict) != 0:
                uni_idx = max(list(UniquePackageDict.keys())) + 1
            else:
                uni_idx = 0
            UniquePackageDict[uni_idx] = separated_package
            InvUniquePackageDict[separated_package] = uni_idx
            
        if remaining_package not in UniquePackageDict.values():
            if len(UniquePackageDict) != 0:
                uni_idx = max(list(UniquePackageDict.keys())) + 1
            else:
                uni_idx = 0
            UniquePackageDict[uni_idx] = remaining_package
            InvUniquePackageDict[remaining_package] = uni_idx
            
        # Convert the modified package into array for each batch.
        self.HomeNode.add_package([0.0, separated_package])
        self.Instruct["sort_node"].add_package([0.0, remaining_package])
            
        return UniquePackageDict, InvUniquePackageDict


    def separate(self, TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, CreationDate):
        """Splits packages by a specified mass, creating a new separated
        package and a remaining package.
    
        This method removes a collection of packages, calculates their total
        mass, and then creates two new packages based on a user-specified mass
        to be moved. The newly created packages are then distributed to the
        home node and a remainder node.
    
        Parameters
        ----------
        TransferNodes : list of Node
            A list of the source Node objects.
        AvailablePackages : list of tuple
            A list of tuples, each containing the age and Package object.
        NodePackageContributions : list of int
            A list tracking the cumulative number of packages from each
            transfer node.
        stepsize : float
            The duration of a single simulation time step.
        timesteps : int
            The total number of time steps in the simulation.
        UniquePackageDict : dict
            A dictionary mapping unique package IDs to their objects.
        InvUniquePackageDict : dict
            An inverse dictionary mapping Package objects to their IDs.
        CreationDate : float
            The current simulation time, which becomes the creation date for
            new packages.
    
        Returns
        -------
        tuple
            A tuple containing the updated UniquePackageDict and
            InvUniquePackageDict.
        """
        
        InputNumber = len(AvailablePackages)

        CombinationStates = []
        TotalMass = 0
        # First create a list of states to be combined.
        if InputNumber != 0:
            mag_idx = 0
            combined_label = ""
            for mag in range(InputNumber):
                time, selected_package = AvailablePackages[0]
                CombinationStates.append([time, selected_package])
                TotalMass += selected_package.get_mass()
                combined_label += selected_package.label + "_"
                
                # Remove state from old node.
                if mag < NodePackageContributions[mag_idx]:
                    TransferNodes[mag_idx].delete_package([time, selected_package])
                else:
                    mag_idx += 1
                    TransferNodes[mag_idx].delete_package([time, selected_package])

                AvailablePackages.remove([time, selected_package])
        
        combined_label = combined_label[:-1]
        
        # Refactor coefficients.
        TotalMass = round(TotalMass, 9)
        if TotalMass != 0:
            if self.Instruct["package_mass"] > TotalMass and self.Crumbs:
                MovedMass = TotalMass
            else:
                MovedMass = self.Instruct["package_mass"] * self.NumOfPacksSample

            Sep_Coef = min(MovedMass / TotalMass, 1)
            Rem_Coef = 1 - Sep_Coef
            
            # 1.5 Create proper package object without reading the nuclide data.
            separated_package = cge.CombinedPackage(PackageStates=list(zip(CombinationStates, [Sep_Coef/self.NumOfPacksSample]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict, HalfLifeDict=selected_package.HalfLifeDict, DecayTypeDict=selected_package.DecayTypeDict, CreationDate=CreationDate)
            separated_package.label = f"S{separated_package.id_iter}_"+combined_label
            if separated_package not in UniquePackageDict.values():
                if len(UniquePackageDict) != 0:
                    uni_idx = max(list(UniquePackageDict.keys())) + 1
                else:
                    uni_idx = 0
                UniquePackageDict[uni_idx] = separated_package
                InvUniquePackageDict[separated_package] = uni_idx
            
            if Rem_Coef >= 0:
                remaining_package = cge.CombinedPackage(PackageStates=list(zip(CombinationStates, [Rem_Coef]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict, HalfLifeDict=selected_package.HalfLifeDict, DecayTypeDict=selected_package.DecayTypeDict, CreationDate=CreationDate)
                remaining_package.label = f"R{remaining_package.id_iter}_"+combined_label
                if remaining_package not in UniquePackageDict.values():
                    if len(UniquePackageDict) != 0:
                        uni_idx = max(list(UniquePackageDict.keys())) + 1
                    else:
                        uni_idx = 0
                    UniquePackageDict[uni_idx] = remaining_package
                    InvUniquePackageDict[remaining_package] = uni_idx
                    
                    # Add all of the remaining into last node that was looked at.
                    # Should be changed in the future to properly account for package origins.
                    TransferNodes[mag_idx].add_package([0, remaining_package])
                
            # Convert the modified package into array for each batch.
            for i in range(self.NumOfPacksSample):
                self.HomeNode.add_package([0, separated_package])
            
        return UniquePackageDict, InvUniquePackageDict


    def refactor(self, batches, TempNuclideList, InvNuclideList, InvArray):
        """Standardizes a nuclide inventory array to a master nuclide list.

        This method takes an inventory array and a list of its nuclides and
        transforms it into a new array that conforms to a provided master
        nuclide list. It fills in zero values for any nuclides missing from the
        original array.
    
        Parameters
        ----------
        batches : int
            The number of batches or packages in the inventory data.
        TempNuclideList : list of str
            The master list of nuclides for the output array.
        InvNuclideList : list of str
            The list of nuclides corresponding to the input inventory array.
        InvArray : numpy.ndarray
            The inventory data array.
    
        Returns
        -------
        numpy.ndarray
            A new NumPy array representing the refactored inventory data.
        """
        
        # Given a list of nuclides (TempNuclideList), convert inventory (InvArray) into same-sized array (replace nonexisting nuclides with 0).
        Y_combined = np.zeros((len(TempNuclideList), batches))

        for val_x, val_y in zip(InvNuclideList, InvArray):
            if val_x in TempNuclideList:
                idx = TempNuclideList.index(val_x)
                Y_combined[idx] = val_y
        
        return Y_combined


    def combine_packages(self, TransferNodes, AvailablePackages, NodePackageContributions, stepsize, timesteps, UniquePackageDict, InvUniquePackageDict, CreationDate):
        """Combines multiple packages into a single, new package and then
        duplicates it.
    
        This method removes a specified number of packages from source nodes,
        consolidates their contents into one new 'combined' package, and then
        creates a number of identical copies of this new package. These copies
        are then added to the home node.
    
        Parameters
        ----------
        TransferNodes : list of Node
            A list of the source Node objects.
        AvailablePackages : list of tuple
            A list of tuples, each containing the age and Package object
            selected for transfer.
        NodePackageContributions : list of int
            A list tracking the cumulative number of packages from each
            transfer node.
        stepsize : float
            The duration of a single simulation time step.
        timesteps : int
            The total number of time steps in the simulation.
        UniquePackageDict : dict
            A dictionary mapping unique package IDs to their objects.
        InvUniquePackageDict : dict
            An inverse dictionary mapping Package objects to their IDs.
        CreationDate : float
            The current simulation time, which becomes the creation date for
            new packages.
    
        Returns
        -------
        tuple
            A tuple containing the updated UniquePackageDict and
            InvUniquePackageDict.
        """
        
        if len(AvailablePackages) >= self.MagnitudeSample or self.Crumbs:
            if len(AvailablePackages) < self.MagnitudeSample and self.Crumbs:
                PackConvQuant = self.MagnitudeSample / self.NumOfPacksSample
                Residue    = int(len(AvailablePackages) % PackConvQuant)
                NumOfPacks = int(len(AvailablePackages) // PackConvQuant) + (Residue != 0)
                InputNumber = len(AvailablePackages)
            else:
                InputNumber = self.MagnitudeSample
                NumOfPacks = self.NumOfPacksSample
            
            CombinationStates = []
            # First create a list of states to be combined.
            if InputNumber != 0:
                mag_idx = 0
                combined_label = ""
                for mag in range(InputNumber):
                    time, selected_package = AvailablePackages[0]
                    CombinationStates.append([time, selected_package])
                    combined_label += selected_package.label + "_"
                    
                    # Remove state from old node.
                    if mag < NodePackageContributions[mag_idx]:
                        TransferNodes[mag_idx].delete_package([time, selected_package])
                    else:
                        mag_idx += 1
                        TransferNodes[mag_idx].delete_package([time, selected_package])

                    AvailablePackages.remove([time, selected_package])
            combined_label = combined_label[:-1]
    
        # 1.5 Create proper package object without reading the nuclide data.
        combined_package = cge.CombinedPackage(PackageStates=list(zip(CombinationStates, [1/NumOfPacks]*len(CombinationStates))), NuclideDict=selected_package.NuclideDict, HalfLifeDict=selected_package.HalfLifeDict, DecayTypeDict=selected_package.DecayTypeDict, CreationDate=CreationDate)
        combined_package.label = f"C{combined_package.id_iter}_"+combined_label
        if combined_package not in UniquePackageDict.values():
            if len(UniquePackageDict) != 0:
                uni_idx = max(list(UniquePackageDict.keys())) + 1
            else:
                uni_idx = 0

            UniquePackageDict[uni_idx] = combined_package
            InvUniquePackageDict[combined_package] = uni_idx
            
        for i in range(NumOfPacks):
            # Convert the modified package into array for each batch.
            self.HomeNode.add_package([0, combined_package])
        return UniquePackageDict, InvUniquePackageDict

    
    def check_criterion(self, Criteria, AwayNode, centile, selected_package_states=None):
        """Checks if a package or group of packages meets a single criterion.
    
        This function evaluates packages based on a single criterion, which can
        be a simple property like mass or a more complex one involving specific
        nuclides, half-life, or decay modes. It is used as a helper function
        for check_criteria.
    
        Parameters
        ----------
        Criteria : dict
            A dictionary containing a single criterion for evaluation.
        AwayNode : Node
            The node object whose packages are being evaluated.
        centile : int
            A percentile value used to filter packages.
        selected_package_states : list of tuple, optional
            A list of tuples with package age and object. Defaults to None.
    
        Returns
        -------
        bool
            True if the selected packages meet the criterion, False otherwise.
        """
        
        # Check if there is enough packages for transfer.
        if len(AwayNode.PackageList) < self.MagnitudeSample and not self.Crumbs:
            return False
        elif len(AwayNode.PackageList) == 0:
            return False
        # Check if there are any criteria present.
        if Criteria == None:
            return True
        
        # copy of transmutable dict.
        Criteria = copy.deepcopy(Criteria)
        
        ### Recalculate the nuclide limit list if decay mode is specified.
        if "nuclide" in Criteria:
            time, selected_package = selected_package_states[0]
            
            NUCLI, CRITE = self.recalculate_composition(selected_package, Criteria["nuclide"], Criteria["criteria"])
            if len(NUCLI) == 0:
                return True
            else:
                Criteria["nuclide"], Criteria["criteria"] = [NUCLI, CRITE]
    
        if (Criteria["variable"] in ["activity_mix", "activity_concentration_mix"]
            and "limits" not in Criteria):
            Criteria["limits"] = dict(zip(Criteria["nuclide"], [1]*len(Criteria["nuclide"])))
        elif (Criteria["variable"] in ["activity_mix", "activity_concentration_mix"]
            and "nuclide" in Criteria and "limits" in Criteria):
            NewLimits = {}
            for nuc in Criteria["nuclide"]:
                if nuc in Criteria["limits"]:
                    NewLimits[nuc] = Criteria["limits"][nuc]
            Criteria["limits"] = NewLimits
    
        # Calculate the total mass of node packages for separation and transfer.
        sep_coef = 1
        if self.Mode == "separate" and Criteria["region"] == "package":
            MovedMass = self.Instruct["package_mass"] * self.NumOfPacksSample
            TotalMass = 0
            for (time, selected_package) in selected_package_states: 
                TotalMass += selected_package.get_mass()
            TotalMass = round(TotalMass, 9)
    
            if TotalMass != 0:
                sep_coef = min(MovedMass / TotalMass, 1)
        
        # If packages are combined/separated, calculate variable value for combined/separated packages.
        if ((self.Mode == "combine" and Criteria["region"] == "package")
            or (self.Mode == "separate" and Criteria["region"] == "package")
            or (self.Mode == "batch" and Criteria["region"] == "package")):
            ComparedVar = 0
            TotalQuantity = 0
            for (time, selected_package) in selected_package_states:
                if Criteria["variable"] == "mass":
                    ComparedVar += selected_package.get_mass()
                elif Criteria["variable"] == "volume":
                    ComparedVar += selected_package.get_volume()
                elif Criteria["variable"] == "inventory":
                    ComparedVar += selected_package.get_inventory(time)
                elif Criteria["variable"] in ["activity", "dose", "activity_concentration"]:
                    ComparedVar += selected_package.get_activity(time)
                    if Criteria["variable"] == "activity_concentration":
                        TotalQuantity += selected_package.get_mass()
                elif Criteria["variable"] == "heat":
                    ComparedVar += selected_package.get_heat(time)
                    if Criteria["variable"] == "heat_concentration":
                        TotalQuantity += selected_package.get_volume()
    
            ComparedVar = ComparedVar * sep_coef / self.NumOfPacksSample
            if Criteria["variable"] in ["activity_concentration", "heat_concentration"]:
                ComparedVar /= TotalQuantity
        
        if Criteria["region"] == "node":
            # By default the node-region criterion is evaluated against AwayNode -- the
            # candidate source node currently being considered for this transfer -- exactly
            # as before. If the criterion specifies its own target node (Criteria["node"],
            # resolved to an actual Node object at XML-parse time), evaluate against that
            # node instead, regardless of what node this order is otherwise working with.
            TargetNode = Criteria.get("node", AwayNode)
            if "nuclide" not in Criteria:
                if Criteria["variable"] == "mass":
                    _, node_var = TargetNode.get_mass()
                elif Criteria["variable"] == "volume":
                    _, node_var = TargetNode.get_volume()
                elif Criteria["variable"] == "inventory":
                    _, node_var = TargetNode.get_inventory()
                elif Criteria["variable"] == "activity":
                    _, node_var = TargetNode.get_activity()
                elif Criteria["variable"] == "activity_concentration":
                    _, node_var = TargetNode.get_activity() / TargetNode.get_mass()
                elif Criteria["variable"] == "heat":
                    _, node_var = TargetNode.get_heat()
                elif Criteria["variable"] == "heat_concentration":
                    _, node_var = TargetNode.get_heat() / TargetNode.get_volume()
                if _ is not None:
                    node_var = node_var.sum(axis=0)
    
                if ((Criteria["principle"] == "max" and np.percentile(node_var, centile) < Criteria["criteria"])
                    or (Criteria["principle"] == "min" and np.percentile(node_var, centile) > Criteria["criteria"])):
                    return False
            else:
                if Criteria["variable"] == "inventory":
                    NuclideDict, node_var = TargetNode.get_inventory()
                elif Criteria["variable"] == "activity":
                    NuclideDict, node_var = TargetNode.get_activity()
                elif Criteria["variable"] == "activity_concentration":
                    NuclideDict, node_var = TargetNode.get_activity() / TargetNode.get_mass()
                elif Criteria["variable"] == "heat":
                    NuclideDict, node_var = TargetNode.get_heat()
                elif Criteria["variable"] == "heat_concentration":
                    NuclideDict, node_var = TargetNode.get_heat() / TargetNode.get_volume()
    
                for nuclide, crit in zip(Criteria["nuclide"], Criteria["criteria"]):
                    if isinstance(nuclide, str):
                        nuclides = [nuclide]
                    else:
                        nuclides = nuclide
                    ComparedVar = 0
                    for nuclide in nuclides:
                        if nuclide in NuclideDict:
                            ComparedVar += node_var[NuclideDict[nuclide]]
    
                    if ((Criteria["principle"] == "max" and np.percentile(ComparedVar, centile) < Criteria["criteria"])
                        or (Criteria["principle"] == "min" and np.percentile(ComparedVar, centile) > Criteria["criteria"])):
                        return False
    
        if Criteria["region"] == "package":
            if "nuclide" not in Criteria and "dose_coefficients" not in Criteria and "limits" not in Criteria:
                if self.Mode == "combine" or self.Mode == "separate" or self.Mode == "batch":
                    if ((Criteria["principle"] == "max" and np.percentile(ComparedVar.sum(axis=0), centile) < Criteria["criteria"])
                        or (Criteria["principle"] == "min" and np.percentile(ComparedVar.sum(axis=0), centile) > Criteria["criteria"])):
                        return False
                else:
                    for (time, selected_package) in selected_package_states:   
                        if Criteria["variable"] == "mass":
                            pack_var = selected_package.get_mass()
                        elif Criteria["variable"] == "volume":
                            pack_var = selected_package.get_volume()
                        elif Criteria["variable"] == "inventory":
                            pack_var = selected_package.get_inventory(time).sum(axis=0)
                        elif Criteria["variable"] == "activity":
                            pack_var = selected_package.get_activity(time).sum(axis=0)
                        elif Criteria["variable"] == "activity_concentration":
                            pack_var = selected_package.get_activity(time).sum(axis=0) / selected_package.get_mass()
                        elif Criteria["variable"] == "heat":
                            pack_var = selected_package.get_heat(time).sum(axis=0)
                        elif Criteria["variable"] == "heat_concentration":
                            pack_var = selected_package.get_heat(time).sum(axis=0) / selected_package.get_volume()
                        
                        if ((Criteria["principle"] == "max" and np.percentile(pack_var, centile) < Criteria["criteria"])
                            or (Criteria["principle"] == "min" and np.percentile(pack_var, centile) > Criteria["criteria"])):
                            return False
            else:
                sum_var = 0
                if "dose_coefficients" in Criteria:
                    radionuclide_list = list(Criteria["dose_coefficients"].keys())
                elif "limits" in Criteria:
                    radionuclide_list = list(Criteria["limits"].keys())
                else:
                    radionuclide_list = Criteria["nuclide"]
    
                for cidx, nuc in enumerate(radionuclide_list):
                    if Criteria["variable"] in ["dose", "activity_mix", "activity_concentration_mix"]:
                        crit = Criteria["criteria"]
                    else:
                        crit = Criteria["criteria"][cidx]
    
                    if self.Mode == "combine" or self.Mode == "separate" or self.Mode == "batch":
                        if isinstance(nuc, str):
                            nuclides = [nuc]
                        else:
                            nuclides = nuc
                        pack_var = 0
                        for nuclide in nuclides:
                            if nuclide in selected_package.NuclideDict:
                                if Criteria["variable"] == "dose":
                                    pack_var += (ComparedVar[selected_package.NuclideDict[nuclide]]
                                                 * Criteria["dose_coefficients"][nuclide])
                                elif Criteria["variable"] in ["activity_mix", "activity_concentration_mix"]:
                                    pack_var += (ComparedVar[selected_package.NuclideDict[nuclide]]
                                                 / Criteria["limits"][nuclide])
                                else:
                                    pack_var += ComparedVar[selected_package.NuclideDict[nuclide]]
    
                        if (Criteria["variable"] not in ["dose", "activity_mix", "activity_concentration_mix"]
                            and ((Criteria["principle"] == "max" and np.percentile(pack_var, centile) < crit)
                            or (Criteria["principle"] == "min" and np.percentile(pack_var, centile) > crit))):
                            return False
    
                        elif Criteria["variable"] in ["dose", "activity_mix", "activity_concentration_mix"]:
                            sum_var += abs(pack_var)
                        
                    else:
                        for (time, selected_package) in selected_package_states:
                            if isinstance(nuc, str):
                                nuclides = [nuc]
                            else:
                                nuclides = nuc
                            pack_var = 0
                            for nuclide in nuclides:
                                if nuclide in selected_package.NuclideDict:
                                    if Criteria["variable"] == "inventory":
                                        pack_var += selected_package.get_inventory(time)[selected_package.NuclideDict[nuclide]]
                                    elif Criteria["variable"] == "activity":
                                        pack_var += selected_package.get_activity(time)[selected_package.NuclideDict[nuclide]]
                                    elif Criteria["variable"] == "activity_concentration":
                                        pack_var += selected_package.get_activity(time)[selected_package.NuclideDict[nuclide]] / selected_package.get_mass()
                                    elif Criteria["variable"] == "heat":
                                        pack_var += selected_package.get_heat(time)[selected_package.NuclideDict[nuclide]]
                                    elif Criteria["variable"] == "heat_concentration":
                                        pack_var += selected_package.get_heat(time)[selected_package.NuclideDict[nuclide]] / selected_package.get_volume()
                                    elif Criteria["variable"] == "dose":
                                        pack_var += (selected_package.get_activity(time)[selected_package.NuclideDict[nuclide]]
                                                    * Criteria["dose_coefficients"][nuclide])
                                    elif Criteria["variable"] == "activity_mix":
                                        pack_var += (selected_package.get_activity(time)[selected_package.NuclideDict[nuclide]]
                                                    / Criteria["limits"][nuclide])
                                    elif Criteria["variable"] == "activity_concentration_mix":
                                        #print(selected_package.get_activity(time), selected_package.get_mass())
                                        pack_var += (selected_package.get_activity(time)[selected_package.NuclideDict[nuclide]] / selected_package.get_mass()
                                                    / Criteria["limits"][nuclide])
                            
    
                            if (Criteria["variable"] not in ["dose", "activity_mix", "activity_concentration_mix"]
                                and ((Criteria["principle"] == "max" and np.percentile(pack_var, centile) < crit)
                                or (Criteria["principle"] == "min" and np.percentile(pack_var, centile) > crit))):
                                return False
    
                            elif Criteria["variable"] in ["dose", "activity_mix", "activity_concentration_mix"]:
                                sum_var += abs(pack_var)
                if (Criteria["variable"] in ["dose", "activity_mix", "activity_concentration_mix"]
                    and ((Criteria["principle"] == "max" and np.percentile(sum_var, centile) < crit)
                    or (Criteria["principle"] == "min" and np.percentile(sum_var, centile) >= crit))):
                    return False
    
        # Return True or False whether it passes or not.
        return True


    def check_criteria(self, AwayNode, centile): 
        """Evaluates whether packages within a given node meet a predefined set
        of criteria.
    
        This method is used to filter packages based on a combination of
        criteria defined in self.Criteria. It can evaluate packages
        individually or as a group, depending on the transfer mode.
    
        Parameters
        ----------
        AwayNode : Node
            The node object whose packages are being evaluated.
        centile : int
            A percentile value used to filter packages.
    
        Returns
        -------
        list of bool
            A list of boolean values indicating whether each package meets all
            specified criteria.
        """

        def check_extra_criteria(criteria_results, additional_success=None):
            result = []
            for values in zip(*criteria_results):  # Iterate column-wise
                if all(values):  
                    result.append(True)  # Case 1: All True
                    continue
                # Check additional success conditions
                if additional_success and list(values) in additional_success:
                    result.append(True)
                    continue
                result.append(False)  # Default to False if no condition met
            
            return result
        
        CriteriaResults = []
        if self.Criteria != None:
            for Crit in self.Criteria:
                if self.Direction == "all" and not (self.Mode == "combine" or self.Mode == "separate" or self.Mode == "batch"):
                    PackageFlags = []
                    for selected_package_state in AwayNode.PackageList:
                        CritResult = self.check_criterion(Crit, AwayNode, centile, [selected_package_state])
                        PackageFlags.append(CritResult)                        
                    CriteriaResults.append(PackageFlags)
                else:
                    if len(AwayNode.PackageList) < self.MagnitudeSample and self.Crumbs:
                        SPidx = len(AwayNode.PackageList)
                    else:
                        SPidx = self.MagnitudeSample

                    if self.Direction == "new":
                        selected_package_states = AwayNode.PackageList[:SPidx]
                    elif self.Direction == "old":
                        selected_package_states = AwayNode.PackageList[-SPidx:]
                    elif self.Direction == "mix":
                        NewPackages = AwayNode.PackageList[:SPidx]
                        OldPackages = AwayNode.PackageList[-SPidx:]
                        PackCount = len(OldPackages)
                        NewPackCount = int(round(PackCount * (1 - self.MixRatio)))
                        OldPackCount = PackCount - NewPackCount
                        selected_package_states = OldPackages[OldPackCount:] + NewPackages[NewPackCount:]
                        
                    PackageFlags = []
                    if self.Mode == "combine" or self.Mode == "separate" or self.Mode == "batch":
                        CritResult = self.check_criterion(Crit, AwayNode, centile, selected_package_states)
                        PackageFlags.append(CritResult)
                        CriteriaResults.append(PackageFlags)
                    else:
                        for selected_package_state in selected_package_states:
                            CritResult = self.check_criterion(Crit, AwayNode, centile, [selected_package_state])
                            PackageFlags.append(CritResult)
                        CriteriaResults.append(PackageFlags)
            
            return check_extra_criteria(CriteriaResults, self.extra_criteria)

        else:
            return [self.check_criterion(self.Criteria, AwayNode, centile)]*len(AwayNode.PackageList)

    def recalculate_composition(self, selected_package, nuclide_list, criteria_list):
        """
        Dynamically recalculates a nuclide list and a criteria list based on
        criteria.
    
        This function expands a list containing half-life or decay-mode
        criteria into a definitive list of actual nuclide names that meet those
        criteria. This is essential for ensuring that subsequent calculations
        use a precise list of nuclides and their corresponding criteria.
    
        Parameters
        ----------
        selected_package : Package
            The package object containing nuclide half-life and decay
            information.
        nuclide_list : list of str
            The initial list of nuclides, which may include criteria strings
            (e.g., "-100y", "alpha").
        criteria_list : list or any
            The value(s) corresponding to each nuclide or criterion.
    
        Returns
        -------
        tuple
            A tuple containing the updated nuclide list and the corresponding
            criteria list.
        """
    
        def _criteria_nuclides(selected_package, nuc):
            if ("-" in nuc or "+" in nuc) and any(char.isdigit() for char in nuc):
                hf_lim = float(nuc[1:-1])
                time_units = nuc[-1]
                if time_units in ["y", "a"]:   
                   hf_lim = hf_lim * (60 * 60 * 24 * 365)
                if time_units in ["m"]:   
                   hf_lim = hf_lim * (60 * 60 * 24 * 365 / 12)
                elif time_units in ["d"]:
                    hf_lim = hf_lim * (60 * 60 * 24)
                elif time_units in ["h"]:
                    hf_lim = hf_lim * (60 * 60)
                
                TempNucList = []
                for mom_nuc, hf in selected_package.HalfLifeDict.items():
                    if "-" in nuc:
                        if hf < hf_lim:
                            TempNucList.append(mom_nuc)
                    elif "+" in nuc:
                        if hf > hf_lim:
                            TempNucList.append(mom_nuc)
            
            elif not any(char.isdigit() for char in nuc):
                TempNucList = []
                for mom_nuc, mode in selected_package.DecayTypeDict.items(): # Need to make it work with combined packages as well.
                    if nuc in mode:
                        TempNucList.append(mom_nuc)

            return TempNucList


        ### Recalculate the nuclide limit list if decay mode is specified.
        for idx, nuc in enumerate(nuclide_list):        
            if ("-" in nuc or "+" in nuc) or (not any(char.isdigit() for char in nuc)):                    
                if "," in nuc:
                    NucCandidatesList = []
                    for nuclide in nuc.split(","):
                        NucCandidatesList.append(_criteria_nuclides(selected_package, nuclide))
                    TempNucList = list(set(NucCandidatesList[0]).intersection(*NucCandidatesList[1:]))
                else:
                    TempNucList = _criteria_nuclides(selected_package, nuc)

                TempNewLabels = []
                TempNewCriteria = []

                if isinstance(criteria_list, list):
                    for item, value in zip(nuclide_list, criteria_list):
                        if item == nuc:
                            TempNewLabels.extend(TempNucList)  # Replace target with all elements from list2
                            TempNewCriteria.extend([value] * len(TempNucList))  # Duplicate value for each replacement
                        else:
                            TempNewLabels.append(item)
                            TempNewCriteria.append(value)
                else:
                    for item in nuclide_list:
                        if item == nuc:
                            TempNewLabels.extend(TempNucList)  # Replace target with all elements from list2
                        else:
                            TempNewLabels.append(item)
                    TempNewCriteria = criteria_list
                    
                nuclide_list = TempNewLabels
                criteria_list = TempNewCriteria
            else:
                TempNewLabels = nuclide_list
                TempNewCriteria = criteria_list
            
        return TempNewLabels, TempNewCriteria


    def initialize(self, initial_nuclides):
        """Identifies all decay products from a set of initial nuclides.

        This method reads a decay chain XML file and recursively finds all
        daughter nuclides for a given list of initial nuclides, providing a
        comprehensive list of all nuclides that will be present in the
        simulation.
    
        Parameters
        ----------
        initial_nuclides : list of str
            A list of the starting nuclide names.
    
        Returns
        -------
        set
            A set of all unique nuclides in the decay chains of the initial
            nuclides.
        """

        # Construct the absolute path to the data file, and get the cached (parse-once)
        # name -> element index instead of re-parsing and linearly re-scanning the whole
        # file for every call and every nuclide in the chain.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, '..', 'data', 'decay_chains_endfb71.xml')
        decay_index = _get_decay_chain_index(file_path)

        # Cache of already-resolved descendant chains for this file, keyed by nuclide
        # name. A nuclide's full set of daughters/granddaughters/etc. never changes, so
        # once "U235" (say) has been resolved once, later calls -- from this Order or any
        # other -- reuse the flattened result instead of re-recursing through the tree.
        descendant_cache = _DECAY_DESCENDANTS_CACHE.setdefault(file_path, {})

        def _resolve_descendants(Parent):
            # Returns the flattened list of every nuclide contributed by Parent's
            # subtree (in the same order the original recursion would append them),
            # or None if Parent isn't in the reference data at all (mirrors the
            # original function's implicit-None fallthrough for that case).
            if Parent in descendant_cache:
                return descendant_cache[Parent]

            parent = decay_index.get(Parent)
            if parent is None:
                descendant_cache[Parent] = None
                return None

            parent_name = parent.attrib["name"]
            if "half_life" not in parent.attrib:
                descendant_cache[Parent] = []
                return []

            CandidateList = []
            for daughter in parent.iter('decay'):
                daughter_name = daughter.attrib["target"]
                if daughter_name != parent_name:
                    CandidateList.append(daughter_name)

            resolved = []
            for candidate in CandidateList:
                sub = _resolve_descendants(candidate)
                if sub is None:
                    descendant_cache[Parent] = None
                    return None
                resolved.extend(sub)
                resolved.append(candidate)

            descendant_cache[Parent] = resolved
            return resolved

        def _find_decay_products(Parent, History):
            # Find decay products for a given nuclide by reusing (or building) its
            # cached resolved subtree, then applying it to this call's History list.
            resolved = _resolve_descendants(Parent)
            if resolved is None:
                return None
            History.extend(resolved)
            return History

        # Collecting daughter nuclides from initial nuclides
        DaughterNuclides = []
        for nuclide in initial_nuclides:
            DaughterNuclides += _find_decay_products(nuclide, list(initial_nuclides))
        
        # Generating a set of simulated nuclides based on the decay chains found (filter out duplicates)
        SimulatedNuclides = set(DaughterNuclides)
        
        return SimulatedNuclides
