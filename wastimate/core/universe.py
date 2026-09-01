# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""
import os
import copy
import json
import numpy as np
from tqdm import tqdm

from wastimate.core import combinedpackage as cge
from wastimate.core import node as de
from wastimate.core import source as rce
from wastimate.core import order as der
from wastimate.core import sim as sim

from wastimate.utils import sir_model as el
from wastimate.utils import annealing_algorithm as aa
from wastimate.utils import sorting_algorithm as hm

class Universe:
    def __init__(self, nodes=None, orders=None, sources=None, history=None, stepsize=None):
        """Initializes the Universe object with nodes, orders, sources, and
        other simulation parameters.

        This is the constructor for the main simulation class. It sets up the
        initial state of the simulation, including the relationships between
        nodes and the loading of essential external data files required for
        calculations.

        Parameters
        ----------
        nodes : list, optional
            A list of Node objects representing the physical locations in the
            simulation. The default is None.
        orders : list, optional
            A list of Order objects that define the transfer rules between
            nodes. The default is None.
        sources : list, optional
            A list of Source objects that introduce new packages into the
            universe.The default is None.
        history : list, optional
            A list to store the history of the simulation's state.
            The default is None.
        stepsize : int or float, optional
            The size of a single time step in the simulation, measured in
            seconds. The default is one year (1 * 60 * 60 * 24 * 365 seconds).

        Returns
        -------
        None
        """

        if nodes is None:
            self.Nodes = []
        else:
            self.Nodes = nodes
            
        if orders is None:
            self.Orders = []
        else:
            self.Orders = orders
            
        if sources is None:
            self.Sources = []
        else:
            self.Sources = sources

        if history is None:
            self.History = []
            if self.Nodes is not None:
                for node in self.Nodes:
                    self.History.append(node)
        else:
            self.History = history
        
        if stepsize is None:
            self.stepsize = 1*60*60*24*365
        else:
            self.stepsize = stepsize
        
        self.NodeDict = {}
        
        self.SimTimeHistory = [0.0]
        self.SimTime = 0.0
        self.batches = 0
        
        self.BasePackageList = []
        self.UniquePackageDict = {}
        self.InvUniquePackageDict = {}
        self.NodePackageHistory = []
        self.HistoryIdx = {}
        self.PackageLabelDict = {}

        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir, '..', 'data', 'mobility_coefficients.json'), "r") as file:
            self.MobilityDict = json.load(file)
            
        with open(os.path.join(script_dir, '..', 'data', 'ingestion_coefficients.json'), "r") as file:
            self.IngestionDict = json.load(file)
            
        with open(os.path.join(script_dir, '..', 'data', 'inhalation_coefficients.json'), "r") as file:
            self.InhalationDict = json.load(file)
        
        with open(os.path.join(script_dir, '..', 'data', 'emission_coefficients.json'), "r") as file:
            self.EmissionDict = json.load(file)


    def __add__(self, Obj):
        """Overloads the addition operator (+) to add Node, Order, or Source
        objects to the Universe.

        This method provides a convenient way to add the main components of the
        simulation to the Universe object.

        Parameters
        ----------
        Obj : Node, Order, or Source
            The object to be added to the simulation.

        Returns
        -------
        Universe
            The Universe object, allowing for method chaining.
        """

        if isinstance(Obj, de.Node):
            if Obj not in self.Nodes:
                self.Nodes.append(Obj)
                self.History.append(Obj)
                
        if isinstance(Obj, der.Order):
            if Obj not in self.Orders:
                self.Orders.append(Obj)
                
        if isinstance(Obj, rce.Source):
            if Obj not in self.Sources:
                self.Sources.append(Obj)
        
        return self


    def __sub__(self, Obj):
        """Overloads the subtraction operator (-) to remove Node, Order, or
        Source objects from the Universe.

        This method removes a specified component from its corresponding list
        within the Universe object.

        Parameters
        ----------
        Obj : Node, Order, or Source
            The object to be removed from the simulation.

        Returns
        -------
        Universe
            The Universe object, allowing for method chaining.
        """

        if isinstance(Obj, de.Node):
            self.Nodes.remove(Obj)
            self.History.remove(Obj)
        
        if isinstance(Obj, der.Order):
            self.Orders.remove(Obj)
            
        if isinstance(Obj, rce.Source):
            self.Sources.remove(Obj)
        
        return self  
    
    
    def initialize(self, timesteps, solver, num_of_cores):
        """Prepares the simulation for execution by initializing and
        refactoring internal data structures, and performing initial decay
        calculations.

        This method finds all unique nuclides and packages, restructures the
        data to ensure consistency across the simulation, and runs a parallel
        simulation of radioactive decay on the base packages. It also warns the
        user if any nuclides lack necessary data (e.g., ingestion coefficients).

        Parameters
        ----------
        timesteps : int
            The number of time steps to simulate.
        solver : str
            The name of the numerical solver to be used (e.g., "cram16").
        num_of_cores : int
            The number of CPU cores to use for parallel processing.

        Returns
        -------
        None
        """

        # Unique radionuclides present in the packages.
        UniqueNuclides = []
        TotalUniquePackages = []
        
        for nidx, node in enumerate(self.Nodes):
            # Number and label the nodes.
            self.NodeDict[node] = nidx
            if len(node.PackageList) != 0:
                # Regular packages
                UniquePackages = list(map(list, zip(*node.PackageList)))[1]
                for Pack in UniquePackages:
                    if isinstance(Pack, cge.CombinedPackage):
                        rootpackages = [x[0][1] for x in Pack.PackageStates]
                        UniquePackages += rootpackages
                TotalUniquePackages += list(set(UniquePackages))
        
        UniquePackages = set([source.Package for source in self.Sources])
        TotalUniquePackages += list(UniquePackages)        
        for UniquePack in TotalUniquePackages:
            if UniquePack not in self.UniquePackageDict.values():
                if len(self.UniquePackageDict) != 0:
                    uni_idx = max(list(self.UniquePackageDict.keys())) + 1
                else:
                    uni_idx = 0
                    
                self.UniquePackageDict[uni_idx] = UniquePack
                self.InvUniquePackageDict[UniquePack] = uni_idx
                self.PackageLabelDict[UniquePack.label] = UniquePack
                
            if not isinstance(UniquePack, cge.CombinedPackage) and UniquePack not in self.BasePackageList:
                self.BasePackageList.append(UniquePack)

        for BasePack in self.BasePackageList:
            UniqueNuclides = UniqueNuclides + list(BasePack.Inventory.keys())
        UniqueNuclides = list(dict.fromkeys(UniqueNuclides))
        
        # Refactor the radionuclide inventory AND bateman matrix.
        # Check whether there are any normal packages.
        TotalHalfLifeDict = {}
        TotalEnergyDict = {}
        for BasePack in self.BasePackageList:
            TotalHalfLifeDict.update(BasePack.HalfLifeDict)
            TotalEnergyDict.update(BasePack.EnergyDict)
            BasePack.refactor(UniqueNuclides)
            
        # Warn the user of non-existing radionuclides in data dictionaries.
        if self.vocal_flag:
            for nuc in UniqueNuclides:
                elementname = nuc.rstrip('0123456789')
                if elementname != elementname.rstrip('_'):
                    elementname = elementname[:-2]
                if nuc not in TotalHalfLifeDict or TotalHalfLifeDict[nuc] != np.inf:
                    IG_flag = nuc not in self.IngestionDict
                    IN_flag = nuc not in self.InhalationDict
                    MO_flag = elementname not in self.MobilityDict
                    
                    if IG_flag or IN_flag or MO_flag:
                        print(f"{nuc} not in " + IG_flag*"ingestion " + IN_flag*"inhalation " + MO_flag*"mobility " + "data.")

        # Calculate radioactive decay, but first check whether there are any normal packages.
        for BasePack in self.BasePackageList:
            BasePack.HalfLifeDict = TotalHalfLifeDict
            BasePack.EnergyDict = TotalEnergyDict
            self.batches = BasePack.batches
        
        if len(BasePack.InventoryStates) == 0:
            self.BasePackageList = sim.parallel_simulate_package(self.BasePackageList, self.stepsize, timesteps, solver, num_of_cores)
        
        elif len(BasePack.InventoryStates) != 0:
            self.BasePackageList = sim.parallel_simulate_package(self.BasePackageList, self.stepsize, timesteps, solver, num_of_cores)
        
        # for BasePack in self.BasePackageList:
        #     BasePack.HalfLifeDict = TotalHalfLifeDict
        #     BasePack.EnergyDict = TotalEnergyDict
                
        #     if len(BasePack.InventoryStates) == 0:
        #         BasePack = parallel_simulate(BasePack, self.stepsize, timesteps, solver, num_of_cores)
            
        #     elif len(BasePack.InventoryStates) != 0:
        #         BasePack = parallel_simulate(BasePack, self.stepsize, timesteps, solver, num_of_cores)
            
        #     self.batches = BasePack.batches
        
        # Go through the sorting orders and create the initial sorting matrix.
        for order in self.Orders:            
            if order.Mode == "Sort":
                sort_nuclides = []
                sort_factors  = []
                for inidx, nuc in enumerate(order.Instruct["sort_nuclides"]):
                    nuc_chain = list(order.initialize([nuc]))
                    sort_nuclides = sort_nuclides + nuc_chain
                    sort_factors = sort_factors + len(nuc_chain) * [order.Instruct["sort_factors"][inidx]]
                
                order.SortFactors = order.refactor(self.batches, UniqueNuclides, sort_nuclides, sort_factors)
      
    
    def simulate(self, timesteps, solver="cram16", num_of_cores=None, vocal=False):
        """
        Runs the full waste management simulation over a specified number of
        time steps.
    
        This method serves as the main simulation loop. It first initializes
        all packages, and, for each time step, it processes transfers (Orders),
        introduces new packages (Sources), ages all existing packages, and
        records the state of the simulation. A progress bar is displayed to 
        show the simulation's progress.
    
        Parameters
        ----------
        timesteps : int
            The number of time steps to run the simulation.
        solver : str, optional
            The numerical solver to use for radioactive decay calculations.
            The default is "cram16".
        num_of_cores : int, optional
            The number of CPU cores to use for parallel processing of decay.
            The default is None.
        vocal : bool, optional
            A flag to enable verbose output, such as warnings about missing
            data. The default is False.
    
        Returns
        -------
        None
            This method does not return a value. It modifies the internal state
            of the Universe object and its components.
        """

        self.vocal_flag = vocal
        # Calculate the radioactive properties of the packages at the very beginning.
        self.initialize(timesteps, solver, num_of_cores)

        # Record save simulation initial history.
        if len(self.History) != 0 and len(self.NodePackageHistory) == 0:            
            TempNodePackageHistory = []
            for idx, node in enumerate(self.History):
                # Create dictionary to retrieve node index from NodePackageHistory.
                self.HistoryIdx[node] = idx
                # Add node state to temporary list.
                TempPackageList = []
                for package_state in node.PackageList:
                    time, package = package_state
                    TempPackageList.append([float(time), self.InvUniquePackageDict[package]])
                        
                # Add node state to temporary list.
                TempNodePackageHistory.append(TempPackageList) #node.PackageList)
            # add node states to history.
            self.NodePackageHistory.append(copy.deepcopy(TempNodePackageHistory))

        # Change the age of the node packages.
        for step in tqdm(range(timesteps), bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', desc="Estimating Waste", colour='red'):   
            self.SimTime += float(self.stepsize)
            for order in self.Orders:
                if order.UntilDelivery <= 0:
                    if order.Direction == "all":
                        (self.UniquePackageDict, self.InvUniquePackageDict) = order.transfer_all(self.stepsize, timesteps, self.UniquePackageDict, self.InvUniquePackageDict, order.Centile, self.SimTime)
                    else:
                        (self.UniquePackageDict, self.InvUniquePackageDict) = order.transfer(self.stepsize, timesteps, self.UniquePackageDict, self.InvUniquePackageDict, order.Centile, self.SimTime)
                    order.UntilDelivery += order.Rate
                else:
                    order.UntilDelivery -= 1

            for source in self.Sources:
                if source.UntilDelivery <= 0:
                        source.transfer()
                        source.UntilDelivery += source.Rate
                else:
                    source.UntilDelivery -= 1
            
            # Update the list of packages and age them.
            for node in self.Nodes:
                # Update the list.
                node.update()
                # Age node package states.
                for sidx in range(len(node.PackageList)):
                    node.PackageList[sidx][0] += float(self.stepsize)
                    
            # Advance simulation time.
            self.SimTimeHistory.append(float(self.SimTime))
            
            # Record save simulation history.
            if len(self.History) != 0:   
                TempNodePackageHistory = []
                for idx, node in enumerate(self.History):
                    TempPackageList = []
                    for package_state in node.PackageList:
                        time, package = package_state
                        TempPackageList.append([time, self.InvUniquePackageDict[package]])
                        
                    # Add node state to temporary list.
                    TempNodePackageHistory.append(TempPackageList) #node.PackageList)
                    
                # add node states to history.
                self.NodePackageHistory.append(copy.deepcopy(TempNodePackageHistory))


    def organize_packages(self, node, variables, size, criteria=None, extra_criteria=None, weights=None, functions=None, optimization=None, indifference=None, preference=None, sort=True, filename="grouped_output.xlsx", timeunit="a"):        
        """Organizes packages within a specified node into groups of a given
        size based on a multi-criteria hazard index, and exports the results to
        an Excel file.
    
        This function calculates a "hazard index" for each package, which is a
        single score representing its overall risk or importance based on a
        combination of different variables (e.g., nuclide inventory, dose rate).
        It then groups packages with similar scores and, if specified, 
        optimizes these groups using a simulated annealing algorithm to meet 
        additional suitability criteria. The final groups are saved to an Excel
        file, and the original package list can be sorted in place.
    
        Parameters
        ----------
        node : Node
            The Node object containing the packages to be organized.
        variables : list of str
            A list of variable names (e.g., "radiotoxicity", "dose_rate") to be
            used for calculating the hazard index.
        size : int
            The desired number of packages in each group.
        criteria : list, optional
            A list of criteria for checking the suitability of the packages.
            The default is None.
        extra_criteria : list, optional
            A list of additional conditions to be used in conjunction with the
            criteria. The default is None.
        weights : list, optional
            A list of weights for each variable in `variables` to adjust their
            importance in the hazard index calculation. The default is None,
            which assigns equal weights.
        functions : list, optional
            A list of utility functions ("linear", "threshold", etc.) to apply
            to each variable. The default is None, which uses a linear function
            for all variables.
        optimization : list, optional
            A list of values indicating whether each variable should be
            maximized (1) or minimized (-1). The default is None, which
            maximizes all variables.
        indifference : list, optional
            A list of indifference thresholds for each variable, used in the
            `Score` function. The default is None.
        preference : list, optional
            A list of preference thresholds for each variable, used in the
            `Score` function. The default is None.
        sort : bool, optional
            If True, the original `node.PackageList` is sorted in place based
            on the calculated hazard index. The default is True.
        filename : str, optional
            The name of the Excel file to which the grouped package information
            will be exported. The default is "grouped_output.xlsx".
        timeunit : str, optional
            The time unit for the age of packages in the output Excel file.
            Supported values include "y" and "a" (years), "d" (days),
            "h" (hours), etc. The default is "a".
    
        Returns
        -------
        None
            The output is an excel file of sorted packages, with on the very
            left a warning if the sorted package does not fulfill WAC. 
            Left column represent package labels and right column package age.
        """
        
        def assess_suitability():
            """Evaluates the suitability of package groups based on defined
            criteria.
        
            This nested helper function iterates through the current grouping
            of packages, applies a set of criteria to each group, and
            identifies which individual packages do not meet the requirements.
            It is a core component of the simulated annealing optimization 
            process.
        
            Parameters
            ----------
            None
                This function operates on variables from the parent function's
                scope, including `current_state`, `statelist`, `TempNode`, and
                `TempOrder`.
        
            Returns
            -------
            list of int
                A list of indices for all packages that failed the suitability
                check.
            """

            SuitabilityList = []
            for group in current_state:
                TempNode.PackageList = [statelist[ind] for ind in group]
                PackageSuitability = TempOrder.check_criteria(TempNode, TempOrder.Centile)
                SuitabilityList.append(PackageSuitability)
            
            SuitabilityFlattened = [item for sublist in SuitabilityList for item in sublist]
            NonSuitableIndices = [i for i, val in enumerate(SuitabilityFlattened) if not val]
            return NonSuitableIndices
        
        if optimization is None:
            optimization = [1]*len(variables)
        if weights is None:
            weights = [1]*len(variables)
        if functions is None:
            functions = ["li"]*len(variables)        

        resultslist = []
        for var in variables:
            varlist = []
            for (age, package), package_label in [[x, x[1].label] for x in node.PackageList]:
                results = self.get_results(packages=package.label, variables=var, times=age)
                varlist.append(results[1])
            resultslist.append(varlist)
        
        labellist = [x[1].label for x in node.PackageList]
        agelist = self.convert_time([x[0] for x in node.PackageList], timeunit, mode="divide")
        statelist = [x for x in node.PackageList]

        TempInput = np.array([np.array(x).flatten() for x in resultslist])
        ValueSets = list(zip(*TempInput))
        
        if indifference is None and preference is None:
            indif_param = [[],[]]
            for vidx in range(len(variables)):
                max_var = max(np.array(resultslist[vidx]).flatten())
                indif_param[0].append(0)
                indif_param[1].append(max_var)
        else:
            indif_param = [indifference, preference]
      
        hazard_index = el.Score(np.array(ValueSets), np.array(indif_param),
                             np.array(optimization), np.array(functions), np.array(weights)) + 1e-3

        grouped_values, grouped_indices = hm.group_sorted_values_with_indices(hazard_index, size)
        flattened_indices = [index for group in grouped_indices for index in group]
        error_idx = []
        
        if criteria is not None:
            TempNode = de.Node()
            TempOrder = der.Order(homenode=TempNode, ordernodes=[TempNode], magnitude=size, mode="batch", crumbs=True, criteria=criteria, extra_criteria=extra_criteria)
            TempOrder.NumOfPacksSample = 1
            
            # Annealing algorithm  
            current_state = [group.copy() for group in grouped_indices]
            temp = 10
            cooling_rate = 0.995
            max_iterations = 1000

            error_idx = assess_suitability()
            current_cost = aa.compute_cost(current_state, error_idx)
            best_state, best_cost, best_error = current_state, current_cost, error_idx

            if len(error_idx) > 0:
                for _ in range(max_iterations):
                    new_state = aa.swap_elements(current_state, error_idx)                        
                    error_idx = assess_suitability()
                    
                    new_cost = aa.compute_cost(new_state, error_idx)
                    cost_diff = new_cost - current_cost
    
                    # Accept better solution or probabilistically accept worse ones
                    if cost_diff < 0 or np.random.rand() < np.exp(-1/temp):
                        current_state, current_cost = new_state, new_cost
                        if new_cost < best_cost:
                            best_state, best_cost, best_error = new_state, new_cost, error_idx
    
                    # Decrease temperature
                    temp *= cooling_rate
                    if temp < 1e-3 or len(error_idx) < 1:
                        break

            grouped_indices = best_state
            error_idx = best_error
        
        grouped_labels = [[labellist[idx] for idx in group] for group in grouped_indices]
        grouped_ages = [[agelist[idx] for idx in group] for group in grouped_indices]
        
        # Calculate the total activity of the packages. 
        grouped_activities = []
        for label_group in grouped_labels:
            t, act, nod, nucl = self.get_results(packages=label_group, variables="activity", nuclides="all")
            grouped_activities.append(list(act[0][0][-1]))
                
        
        # Add empty line to other groups and a nuclide legend for activities.
        grouped_labels.insert(0, [])
        grouped_ages.insert(0, [])
        grouped_activities.insert(0, nucl)
        
        hm.storage_to_excel([grouped_labels], [grouped_ages], list3=[grouped_activities], error_list=error_idx, filename=filename)
        
        # Sort into the new order.
        if sort:
            node.PackageList = [statelist[ind] for ind in flattened_indices]

        
    def organize_node(self, node, variables, size, weights=None, functions=None, optimization=None, indifference=None, preference=None, mode="closure", plot="image", filename="storage_output.xlsx", timeunit="a"):
        """
        Organizes packages within a node into a simulated storage array based
        on a multi-criteria hazard index and exports the results to an Excel
        file and a plot.
    
        This method calculates a single, composite score (hazard index) for
        each package based on a list of variables. It then fills a storage
        array of a specified size with the packages according to this score and 
        the chosen organization mode. The results, including package labels and
        ages, are exported to an Excel file, and the filled storage array can
        be visualized as a 3D plot.
    
        Parameters
        ----------
        node : Node
            The Node object containing the packages to be organized.
        variables : list of str
            A list of variable names (e.g., "radiotoxicity", "dose_rate") to b
            e used for calculating the hazard index.
        size : list of int
            A list specifying the dimensions of the storage container, e.g.,
            [rows, columns, layers].
        weights : list of int or float, optional
            Weights for each variable in `variables` to adjust their importance
            in the hazard index calculation. The default is None, which assigns
            equal weights.
        functions : list of str, optional
            A list of utility functions to apply to each variable (e.g., "li"
            for linear). The default is None, which uses a linear function for
            all variables.
        optimization : list of int, optional
            A list where `1` indicates a variable should be maximized and `-1`
            indicates it should be minimized. The default is None, which
            maximizes all variables.
        indifference : list, optional
            A list of indifference thresholds for each variable, used in the
            `Score` function. The default is None.
        preference : list, optional
            A list of preference thresholds for each variable, used in the
            `Score` function. The default is None.
        mode : str, optional
            The method used to fill the storage container. Supported modes
            include "closure", which fills from the center outwards, and
            "ordered", which fills sequentially. The default is "closure".
        plot : str, optional
            The type of visualization to generate. Options are "image" for a
            static image or "gif" for an animated GIF. The default is "image".
        filename : str, optional
            The name of the Excel file to which the grouped package information
            will be exported. The default is "storage_output.xlsx".
        timeunit : str, optional
            The time unit for the age of packages in the output Excel file.
            Supported values include "y" (years), "a" (annum), "d" (days),
            "h" (hours), etc. The default is "a".
    
        Returns
        -------
        None
            This method does not return a value. It generates a file and/or a
            plot.
        """
        
        if optimization is None:
            optimization = [1]*len(variables)
        if weights is None:
            weights = [1]*len(variables)
        if functions is None:
            functions = ["li"]*len(variables)        

        resultslist = []
        for var in variables:
            varlist = []
            for (age, package), package_label in [[x, x[1].label] for x in node.PackageList]:
                results = self.get_results(packages=package.label, variables=var, times=age)
                varlist.append(results[1])
            resultslist.append(varlist)
        
        labellist = [x[1].label for x in node.PackageList]
        agelist = self.convert_time([x[0] for x in node.PackageList], timeunit, mode="divide")
        statelist = [x for x in node.PackageList]

        TempInput = np.array([np.array(x).flatten() for x in resultslist])
        ValueSets = list(zip(*TempInput))
        
        if indifference is None and preference is None:
            indif_param = [[],[]]
            for vidx in range(len(variables)):
                max_var = max(np.array(resultslist[vidx]).flatten())
                indif_param[0].append(0)
                indif_param[1].append(max_var)
        else:
            indif_param = [indifference, preference]
        
        hazard_index = el.Score(np.array(ValueSets), np.array(indif_param),
                             np.array(optimization), np.array(functions), np.array(weights)) + 1e-3
        
        StorageArray = size
        
        filled_storage, indexed_storage = hm.fill_config(hazard_index, StorageArray, mode)

        labeled_storage = [[[labellist[int(idx)] if not np.isnan(idx) and idx != -1 else "" for idx in row]for row in layer] for layer in indexed_storage.T]
        aged_storage = [[[agelist[int(idx)] if not np.isnan(idx) and idx != -1 else "" for idx in row]for row in layer] for layer in indexed_storage.T]
        
        hm.storage_to_excel([[[item] for item in labellist]], [[[item] for item in hazard_index]], filename="listed_"+filename)
        hm.storage_to_excel(labeled_storage, aged_storage, filename=filename)
        
        if plot == "image":
            hm.plot_3d_array(filled_storage, colormap='coolwarm')
        elif plot == "gif":
            hm.plot_3d_array_gif(filled_storage, colormap='coolwarm')

    
    def convert_time(self, times, timeunit, mode="multiply"):
        """ Convertis time values between different units and a base unit of
        seconds.

        Parameters
        ----------
        times : list
            List or array of numerical values representing time.
        timeunit : str
            A string specifying the unit of the times input. Supported units are:
                "y" or "a" for years
                "m" for months
                "d" for days
                "h" for hours
        mode : str, optional
            optional string parameter controls the direction of the conversion.
            The default is "multiply".
                "multiply" converts the input times from the specified timeunit
                to seconds.
                "divide" converts the input times from seconds to the specified
                timeunit.

        Returns
        -------
        np.array
            returns a NumPy array with the converted time values.
        """
        
        timeunit = timeunit
        times_array = np.array(times)
        if timeunit in ["y", "a"]:   
            factor = 60 * 60 * 24 * 365
        if timeunit in ["m"]:
            factor = 60 * 60 * 24 * 365 / 12
        elif timeunit in ["d"]:
            factor = 60 * 60 * 24
        elif timeunit in ["h"]:
            factor = 60 * 60
        if mode == "multiply":
            return times_array * factor
        if mode == "divide":
            return times_array / factor
        

    def get_results(self, nodes=None, packages=None, variables=None, nuclides=None, times=None, external_file=None, unify=False, mode="sum", timeunit="a"):
        """Retrieves and processes simulation results based on specified
        filters.

        This function is designed to extract, filter, and aggregate various
        simulation data points from the `Universe` object's history. It allows
        for flexible querying of data by nodes, packages, time, and variables
        like inventory, activity, and radiotoxicity.
    
        Parameters
        ----------
        nodes : str or list of str, optional
            A label or a list of labels for the nodes to include in the results.
            If None, all nodes are considered. The default is None.
        packages : str or list of str, optional
            A label or a list of labels for the packages to include. If None,
            all packages within the specified nodes are considered.
            The default is None.
        variables : str or list of str, optional
            A label or a list of labels for the variables to retrieve (e.g.,
            "inventory", "activity", "heat"). The default is None, which
            implies retrieving all variables.
        nuclides : str or list of str, optional
            A label or a list of labels for the nuclides to include. If "all"
            or None, the results for all unique nuclides are returned. 
            The default is None.
        times : float or list of float, optional
            A single time point or a list of time points to retrieve data for.
            If None, the entire simulation history is used. The default is None.
        external_file : dict, optional
            An external dictionary containing data to be used as multipliers
            for the requested results. The keys should be nuclide labels.
            The default is None.
        unify : bool, optional
            If True, the results from all specified nodes and packages are 
            unified into a single data matrix. The default is False.
        mode : str, optional
            The aggregation mode to use when combining data (e.g., "sum",
            "min", "max", "avg"). The default is "sum".
        timeunit : str, optional
            The unit of time for the `times` parameter. Supported values
            include "y" (years), "a" (annum), "d" (days), "h" (hours), etc.
            The default is "a".
    
        Returns
        -------
        tuple
            A tuple containing:
            - recorded_times_array (numpy.ndarray): The time points for which
              data was retrieved.
            - data_matrix (list of numpy.ndarray): The retrieved data organized
              by variables.
            - labels (tuple of list): A tuple containing a list of node labels
              and a
              list of package labels.
            - nuclideslist (list of str): A list of the nuclide labels included
              in the results.
        """

        def modify_data(dataset, axis, mode):
            """Performs an aggregation operation on a NumPy array along a 
            specified axis.
    
            This helper function allows for flexible aggregation of data by
            summing, finding the minimum, maximum, or average along a given
            axis.
    
            Parameters
            ----------
            dataset : numpy.ndarray
                The data array to be modified.
            axis : int or tuple of int
                The axis or axes along which to apply the operation.
            mode : str
                The operation to perform: "sum", "min", "max", or "avg".
    
            Returns
            -------
            numpy.ndarray
                The modified array.
            """

            match mode:
                case "sum":
                    dataset = np.sum(dataset, axis=axis)
                case "min":
                    dataset = np.min(dataset, axis=axis)
                case "max":
                    dataset = np.max(dataset, axis=axis)
                case "avg":
                    dataset = np.mean(dataset, axis=axis)
            return dataset
        
        def get_value(package, variable, time, nuc_indices):        
            """Retrieves a specific value for a given package and variable at a
            certain time.
    
            This helper function handles the logic for retrieving various
            package properties, such as mass, activity, heat, and radiotoxicity,
            by calling the appropriate package methods.
    
            Parameters
            ----------
            package : Package
                The package object from which to retrieve the value.
            variable : str
                The name of the variable to retrieve (e.g., "inventory",
                "heat").
            time : float
                The time at which to get the value.
            nuc_indices : list of int
                A list of indices for the nuclides to consider in the
                calculation.
    
            Returns
            -------
            int, float, or numpy.ndarray
                The calculated value for the requested variable.
            """
        
            match variable:
                case "mass":
                    return package.get_mass()
                case "volume":
                    return package.get_volume()
                case "inventory":
                    return package.get_inventory(time)[indices]
                case "activity":
                    return package.get_activity(time)[indices]
                case "activity_concentration":
                    return package.get_activity(time)[indices]
                case "heat":
                    return package.get_heat(time)[indices]
                case "heat_concentration":
                    return package.get_heat(time)[indices]
                case "halflife":
                    Activity = package.get_activity(time)
                    HalfLifeSum = []
                    for nidx in nuc_indices:
                        HalfLife = package.HalfLifeDict[package.InvNuclideDict[nidx]]
                        if HalfLife < np.inf:
                            HalfLifeSum.append(HalfLife * Activity[nidx])                            
                    return max(HalfLifeSum)
                case "mobility":
                    MobilitySum = []
                    for nidx in nuc_indices:
                        nuc = package.InvNuclideDict[nidx]
                        elementname = nuc.rstrip('0123456789')
                        if elementname != elementname.rstrip('_'):
                            elementname = elementname[:-2]
                        if package.HalfLifeDict[nuc] != np.inf and elementname in self.MobilityDict:
                            ActivityValue = package.get_activity(time)[package.NuclideDict[nuc]]
                            MobilitySum.append(ActivityValue * 1/self.MobilityDict[elementname])
                    return max(MobilitySum)
                case "ingestion_radiotoxicity":
                    ToxicitySum = 0
                    for nidx in nuc_indices:
                        nuc = package.InvNuclideDict[nidx]
                        if package.HalfLifeDict[nuc] != np.inf and nuc in self.IngestionDict:
                            ActivityValue = package.get_activity(time)[package.NuclideDict[nuc]]
                            ToxicitySum += ActivityValue * self.IngestionDict[nuc]
                    return ToxicitySum
                case "inhalation_radiotoxicity":
                    ToxicitySum = 0
                    for nidx in nuc_indices:
                        nuc = package.InvNuclideDict[nidx]
                        if package.HalfLifeDict[nuc] != np.inf and nuc in self.InhalationDict:
                            ActivityValue = package.get_activity(time)[package.NuclideDict[nuc]]
                            ToxicitySum += ActivityValue * self.InhalationDict[nuc]
                    return ToxicitySum
                case "emission":
                    EmissionSum = 0
                    for nidx in nuc_indices:
                        nuc = package.InvNuclideDict[nidx]
                        if package.HalfLifeDict[nuc] != np.inf and nuc in self.EmissionDict:
                            ActivityValue = package.get_activity(time)[package.NuclideDict[nuc]]
                            EmissionSum += ActivityValue * self.EmissionDict[nuc]
                    return EmissionSum
                    

        # Any of the variables could be a list.
        if not isinstance(nodes, list) and nodes is None:
            nodeslist = [x.label for x in self.Nodes]
        elif not isinstance(nodes, list) and nodes is not None:
            nodeslist = [nodes]
        else:
            nodeslist = nodes
                
        if not isinstance(packages, list) and packages is not None:
            packageslist = [packages]
        else:
            packageslist = packages
            
        if not isinstance(variables, list) and variables is not None:
            variableslist = [variables]
        else:
            variableslist = variables

        if not isinstance(nuclides, list):
            if nuclides is None or nuclides == "all":
                nuclideslist = list(self.BasePackageList[0].NuclideDict.keys())
            else:
                nuclideslist = [nuclides]
        else:
            nuclideslist = nuclides
            
        if not isinstance(times, list):
            if times is None:
                times = self.SimTimeHistory
            else:
                times = [times]

        # Convert the input times to simulation times.
        if times is not self.SimTimeHistory:
            times_array = self.convert_time(times, timeunit, mode="multiply")                    
            simtime_array = np.array(self.SimTimeHistory)  
            times = np.array([simtime_array[np.argmin(np.abs(simtime_array - y))] for y in times_array]).tolist()
            
        # Convert nuclides into filterable array indices.
        if external_file is None:
            indices = [self.BasePackageList[0].NuclideDict[x] for x in nuclideslist]
        else:
            indices = [self.BasePackageList[0].NuclideDict[x] for x in nuclideslist if x in external_file]
            json_multiplier = np.array([external_file[self.BasePackageList[0].InvNuclideDict[x]] for x in indices])

        # Convert times into filterable array indices.
        time_indices = np.nonzero(np.isin(self.SimTimeHistory, times))[0]
        
        # Define variables, where nuclides play a role, to form a correct sized matrix.
        nuclide_variables = (["inventory", "activity", "heat",
                              "activity_concentration", "heat_ mass_concentration",
                              "heat_concentration", "activity_volume_concentration",
                              "dose", "ingestion_radiotoxicity", "mobility",
                              "inhalation_radiotoxicity", "emission"])
        
        mass_variables = ["activity_concentration", "heat_mass_concentration"]
        volume_variables = ["heat_concentration", "activity_volume_concentration"]
        # If nodes is defined, take a node-specific approach,
        # looking at the values of the nodes at package ages
        # rather than the individual packages and their decay curves.
        data_matrix = []
        recorded_times = []
        packagelabelslist = []
        packagelabels = []
        nodelabelslist = []
        if nodes is not None:
            for variable in variableslist:
                variable_size = 1 + (variable in nuclide_variables)* (len(nuclideslist) - 1) # If it contains nuclide specific data or not.
                NodeValues = np.zeros((len(nodeslist), len(times), variable_size, self.batches))
                packagelabels = []
                nodelabelslist = []
                for nidx, node in enumerate([x for x in self.Nodes if x.label in nodes]):
                    node_idx = self.HistoryIdx[node]

                    TimeValues = np.zeros((len(times), variable_size, self.batches))
                    for tidx, time_idx in enumerate(time_indices):
                        PackagesInNodeAtTime = self.NodePackageHistory[time_idx][node_idx]
                        node_avg_factor = 0
                        for age, pidx in PackagesInNodeAtTime:
                            pack = self.UniquePackageDict[pidx]
                            if packages != None and pack.label not in packageslist: # Skip packages which are not named
                                continue
                            packagelabels.append(pack.label)
                            
                            if variable in mass_variables:
                                node_avg_factor += pack.get_mass()
                            elif variable in volume_variables:
                                node_avg_factor += pack.get_volume()
                            
                            if external_file is not None:
                                TempValue = get_value(package=pack, variable=variable, time=age, nuc_indices=indices) * json_multiplier
                            else:
                                TempValue = get_value(package=pack, variable=variable, time=age, nuc_indices=indices)
                            TimeValues[tidx,:,:] += TempValue
                        TimeValues[tidx,:,:] /= ((node_avg_factor>0)*(node_avg_factor - 1) + 1)
                    NodeValues[nidx, :, :, :] += TimeValues
                    packagelabelslist.append(list(set(packagelabels)))
                    nodelabelslist.append(node.label)
                # Do the averaging of packages in node separately from other parameters.
                axis_set = tuple([-1] + [-2]*(nuclides is None or len(nuclideslist) == 1)+ [-3]*(len(times) == 1) + [-4]*(unify))
                data_matrix.append(modify_data(NodeValues, axis_set, mode)) # data_matrix[variables, nodes, times, nuclides, batches]

        else:
            if packages is None:
                packagelist = list(self.UniquePackageDict.values())
            else:
                packagelist = [x for x in list(self.UniquePackageDict.values()) if x.label in packageslist]

            for variable in variableslist:
                variable_size = 1 + (variable in nuclide_variables) * (len(nuclideslist) - 1) # If it contains nuclide specific data or not.
                PackageValues = np.zeros((len(packagelist), len(times), variable_size, self.batches))
                for pidx, package in enumerate(packagelist):
                    TimeValues = np.zeros((len(times), variable_size, self.batches))
                    for tidx, time_idx in enumerate(time_indices):
                        # apply nuclide / dictionary logic here.
                        if isinstance(package, cge.CombinedPackage):
                            if self.SimTimeHistory[time_idx] < package.creationdate:
                                TempValue = np.zeros((variable_size, self.batches)) * np.nan
                            else:
                                if external_file is not None:
                                    TempValue = get_value(package=package, variable=variable, time=self.SimTimeHistory[time_idx]-package.creationdate, nuc_indices=indices) * json_multiplier
                                else:
                                    TempValue = get_value(package=package, variable=variable, time=self.SimTimeHistory[time_idx]-package.creationdate, nuc_indices=indices)
                        else:
                            if external_file is not None:
                                TempValue = get_value(package=package, variable=variable, time=self.SimTimeHistory[time_idx], nuc_indices=indices) * json_multiplier
                            else:
                                TempValue = get_value(package=package, variable=variable, time=self.SimTimeHistory[time_idx], nuc_indices=indices)
                        
                        if variable in mass_variables:
                            TimeValues[tidx, :, :] += TempValue / package.get_mass()
                        elif variable in volume_variables:
                            TimeValues[tidx, :, :] += TempValue / package.get_volume()
                        else:
                            TimeValues[tidx, :, :] += TempValue

                    PackageValues[pidx, :, :, :] += TimeValues

                axis_set = tuple([-1] + [-2]*(nuclides is None or len(nuclideslist) == 1) + [-3]*(len(times) == 1) + [-4]*(unify))
                data_matrix.append(modify_data(PackageValues, axis_set, mode)) # data_matrix[variables, packages, times, nuclides, batches]
            
            nodelabelslist = nodeslist
            packagelabelslist = [x.label for x in packagelist]
        # How to handle the summation of activity concentration for example, where we need to use weighed sum?
        for tidx, time_idx in enumerate(time_indices):
            recorded_times.append(self.SimTimeHistory[time_idx])
        recorded_times_array = self.convert_time(recorded_times, timeunit, mode="divide")
        
        if unify:
            nodelabelslist = ["unified node"]; packagelabelslist = ["unified package"]

        return recorded_times_array, data_matrix, tuple((nodelabelslist, packagelabelslist)), nuclideslist