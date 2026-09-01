# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""

import numpy as np
import itertools

class CombinedPackage:
    
    id_iter = itertools.count()
    
    def __init__(self, PackageStates, NuclideDict, HalfLifeDict, DecayTypeDict, CreationDate):
        """Initializes a CombinedPackage object by compiling a list of
        sub-packages.

        This constructor is responsible for creating a single entity from
        multiple individual packages or other CombinedPackages, handling the
        recursive unraveling of nested structures and maintaining consistent
        nuclide data.

        Parameters
        ----------
        PackageStates : list of tuple
            A list where each element is a tuple containing a Package object
            and a division factor for that package.
        NuclideDict : dict
            A dictionary mapping nuclide names to their unique integer indices.
        HalfLifeDict : dict
            A dictionary mapping nuclide names to their half-lives.
        DecayTypeDict : dict
            A dictionary mapping nuclide names to their decay types.
        CreationDate : float
            The creation date of this combined package.
        """

        self.NuclideDict = NuclideDict
        self.HalfLifeDict = HalfLifeDict
        self.DecayTypeDict = DecayTypeDict
        
        self.creationdate = CreationDate
        self.label = "CombinedPackage" + str(next(self.id_iter))
        
        def compile_package_list(PackageStates):
            PackStates = []
            for PackageState, DivFactor in PackageStates:
                time, pack = PackageState
                
                if isinstance(pack, CombinedPackage):
                    compile_package_list(pack.PackageStates)
                    
                    for PackageStatez, DivFactorz in pack.PackageStates:
                        PackStates.append([[PackageStatez[0]+time, PackageStatez[1]], DivFactorz*DivFactor])

                else:
                    PackStates.append([PackageState, DivFactor])
                    
            return PackStates
        
        self.PackageStates = compile_package_list(PackageStates)
        
    def get_mass(self):
        """Calculates the total mass of the combined package.
    
        Returns
        -------
        float
            The sum of the masses of all constituent packages, weighted by
            their division factors.
        """

        TempMass = 0
        for (time, pack), DivFactor in self.PackageStates:
            TempMass += pack.Mass * DivFactor
        return TempMass 
    
    def get_volume(self):
        """Calculates the total volume of the combined package.
    
        Returns
        -------
        float
            The sum of the volumes of all constituent packages, weighted by
            their division factors.
        """

        TempVolume = 0
        for (time, pack), DivFactor in self.PackageStates:
            TempVolume += pack.Volume * DivFactor
        return TempVolume

    def get_inventory(self, age):
        """Calculates the combined nuclide inventory at a specified age.

        Parameters
        ----------
        age : float
            The age in seconds at which to retrieve the inventory data.
    
        Returns
        -------
        numpy.ndarray
            A NumPy array containing the total atomic inventory of all packages
            at the given age.
        """

        TempInventory = 0
        for (time, pack), DivFactor in self.PackageStates:
            if time+age not in pack.InventoryStates:
                return pack.InventoryStates[0] * np.nan
            TempInventory += pack.InventoryStates[time+age] * DivFactor
        return TempInventory

    def get_activity(self, age):
        """Calculates the total activity of the combined package at a specified
        age.

        Parameters
        ----------
        age : float
            The age in seconds at which to retrieve the activity data.
    
        Returns
        -------
        numpy.ndarray
            A NumPy array containing the total activity of all packages at the
            given age.
        """

        TempActivity = 0
        for (time, pack), DivFactor in self.PackageStates:
            if time+age not in pack.ActivityStates:
                return pack.ActivityStates[0] * np.nan
            TempActivity += pack.ActivityStates[time+age] * DivFactor
        return TempActivity

    def get_heat(self, age):
        """Calculates the total decay heat of the combined package at a 
        specified age.
    
        Parameters
        ----------
        age : float
            The age in seconds at which to retrieve the heat data.
    
        Returns
        -------
        numpy.ndarray
            A NumPy array containing the total heat output of all packages at
            the given age.
        """

        TempHeat = 0
        for (time, pack), DivFactor in self.PackageStates:
            if time+age not in pack.HeatStates:
                return pack.HeatStates[0] * np.nan
            TempHeat += pack.HeatStates[time+age] * DivFactor
        return TempHeat

    def get_packages(self):
        """Retrieves all individual Package objects within the combined
        package.
    
        Returns
        -------
        list
            A list of the individual Package objects that were combined.
        """
    
        TempPackages = []
        for i, j in self.PackageStates:
            TempPackages.append(i[1])

        return TempPackages
