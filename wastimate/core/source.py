# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""

import itertools

class Source:
    
    id_iter = itertools.count()
    
    def __init__(self, awaynode, package, magnitude, label=None, rate=0):
        """Initializes a Source object for transferring packages to an away
        node.

        Parameters
        ----------
        awaynode : object
            The Node object to which packages will be transferred.
        package : object
            The Package object to be transferred.
        magnitude : int, float, or scipy.stats object
            The number of packages to transfer, which can be a single value
            or a statistical distribution.
        label : str, optional
            A user-defined label for the source. Defaults to None.
        rate : float, optional
            The rate of transfer in seconds. Defaults to 0.

        Returns
        -------
        None
            This method initializes the object's attributes and does not return a value.
        """

        if label == None:
            self.label = "Source" + str(next(self.id_iter))
        else:
            self.label = label
            
        # Initializes a Source with AwayNode, Package, Magnitude, and an optional Rate.
        self.AwayNode = awaynode
        self.Package = package
        self.Magnitude = magnitude
        
        self.Rate = rate
        self.UntilDelivery = rate

    def transfer(self):
        """Transfers packages from the source to the away node.
    
        This method samples from the magnitude distribution to determine the
        number of packages to transfer and then adds them to the away node's
        package list.
    
        Returns
        -------
        None
            This method modifies the package list of the associated away node
            in-place and does not return a value.
        """
        # Sample from the Magnitude distribution, convert the distribution into a temp int value.
        if isinstance(self.Magnitude, int) or isinstance(self.Magnitude, float):
            self.MagnitudeSample = int(round(self.Magnitude))
        else:
            self.MagnitudeSample = int(self.Magnitude.rvs(1))
        
        # Transfers packages from the source to the away node.
        for mag in range(self.MagnitudeSample):
            PackageState = [0.0, self.Package]
            self.AwayNode.add_package(PackageState)
