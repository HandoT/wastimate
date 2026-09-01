# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""

import itertools
from wastimate.core import package as ge

class Node:
    
    id_iter = itertools.count()
    
    def __init__(self, packagelist=None, label=None, multiplication_factor=1):
        """Initializes a Node to hold a collection of packages.
    
        Parameters
        ----------
        packagelist : list, optional
            A list of Package objects to be added to the node.
            Defaults to None.
        label : str, optional
            A unique label for the node. Defaults to a generated label.
        multiplication_factor : int, optional
            A factor to multiply the initial number of packages by.
            Defaults to 1.
        """
    
        if label == None:
            self.label = "Node" + str(next(self.id_iter))
        else:
            self.label = label
        
        # Initializes a Node with an optional PackageList and multiplication factor.
        if packagelist == None:
            self.PackageList = []
            self.TempPackageList = []
        else:
            self.PackageList  = packagelist
            self.TempPackageList = packagelist
            
        MultiPackageList  = []
        for pack in self.PackageList:
            for i in range(multiplication_factor):
                MultiPackageList.append([0, pack])

        self.PackageList = MultiPackageList
        self.TempPackageList =  MultiPackageList
                    

    def add_package(self, package):
        """Adds a package to the node's temporary package list.

        Parameters
        ----------
        package : list or tuple
            A list or tuple containing a timestamp and a Package object.
    
        Returns
        -------
        None
            This method modifies the self.TempPackageList in-place.
        """

        self.TempPackageList.insert(0, package)
        
    def delete_package(self, package):
        """Deletes a specific package from the node's temporary package list.
    
        Parameters
        ----------
        package : list or tuple
            The package to be removed from the list.
    
        Returns
        -------
        None
            This method modifies the self.TempPackageList in-place.
        """

        self.TempPackageList.remove(package)
        
    def __add__(self, Obj):
        """Overloads the '+' operator to add a package to the node.

        Parameters
        ----------
        Obj : Package or list
            The Package object or a list containing a timestamp and a Package
            object to be added.
    
        Returns
        -------
        Node
            The Node object itself, to allow for chaining of additions.
        """
    
        if isinstance(Obj, ge.Package):
            self.TempPackageList.insert(0, [0.0, Obj])
        else:
            self.TempPackageList.insert(0, Obj)
            
        return self
    
    def update(self):
        """Finalizes changes to the node by updating the official package list.

        Returns
        -------
        None
            This method updates the self.PackageList in-place.
        """

        self.PackageList = self.TempPackageList
        
    def get_packages(self):
        """Retrieves the list of packages and their ages currently in the node.
    
        Returns
        -------
        list
            A list of two lists: the first containing package ages and the 
            second containing the corresponding Package objects.
        """

        return list(map(list, zip(*self.PackageList)))
    
    def get_labels(self):
        """Retrieves the labels of all packages in the node.

        Returns
        -------
        tuple
            A tuple containing None and a list of strings, where each string is
            a package label.
        """
        # Calculates and retrieves the labels of the packages in the node.
        TempLabels = []
        if len(self.PackageList) != 0:
            for pack in list(map(list, zip(*self.PackageList)))[1]:
                TempLabels.append(pack.label)
        return None, TempLabels
    
    def get_package(self, retrieve_label):
        """Retrieves a specific package from the node using its label.

        Parameters
        ----------
        retrieve_label : str
            The label of the package to retrieve.
    
        Returns
        -------
        Package or None
            The matching Package object, or None if the label is not found.
        """

        if len(self.PackageList) != 0:
            for pack in list(map(list, zip(*self.PackageList)))[1]:
                if pack.label == retrieve_label:
                    return pack
    
    def get_mass(self):
        """Calculates and retrieves the total mass of all packages in the node.

        Returns
        -------
        tuple
            A tuple containing None and a float representing the total mass.
        """

        TempMass = 0
        if len(self.PackageList) != 0:
            for pack in list(map(list, zip(*self.PackageList)))[1]:
                TempMass += pack.get_mass()
        return None, TempMass
    
    def get_volume(self):
        """Calculates and retrieves the total volume of all packages in the
        node.
    
        Returns
        -------
        tuple
            A tuple containing None and a float representing the total volume.
        """

        TempVolume = 0
        if len(self.PackageList) != 0:
            for pack in list(map(list, zip(*self.PackageList)))[1]:
                TempVolume += pack.get_volume()
        return None, TempVolume
    
    def get_heat(self):
        """Retrieves the total decay heat generated by all packages in the
        node.

        Returns
        -------
        tuple
            A tuple containing the nuclide dictionary and a NumPy array of the
            total heat output for each nuclide.
        """

        TempHeat = 0
        for (time, pack) in self.PackageList:
            TempHeat += pack.get_heat(time)
        return pack.NuclideDict, TempHeat
    
    def get_inventory(self):
        """Retrieves the total inventory of nuclides across all packages in the
        node.

        Returns
        -------
        tuple
            A tuple containing the nuclide dictionary and a NumPy array of the
            total nuclide inventory.
        """

        TempInventory = 0
        for (time, pack) in self.PackageList:
            TempInventory += pack.get_inventory(time)
        return pack.NuclideDict, TempInventory
    
    def get_activity(self):
        """Retrieves the total activity of nuclides across all packages in the
        node.

        Returns
        -------
        tuple
            A tuple containing the nuclide dictionary and a NumPy array of the
            total nuclide activity.
        """

        # Retrieves the total activity of nuclides in the node.
        TempActivity = 0
        for (time, pack) in self.PackageList:
            TempActivity += pack.get_inventory(time)
                
        return pack.NuclideDict, TempActivity