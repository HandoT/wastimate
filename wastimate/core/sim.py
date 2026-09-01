# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:16:40 2023

@author: hando
"""

import copy
import numpy as np
import multiprocessing

def parallel_calculate_states(arguments):
    """Performs a single-package simulation for a subset of batches.

    This is a helper function designed for parallel execution via
    multiprocessing. It takes a Package object and simulation parameters and
    calculates the nuclide states for a specific range of batches.

    Parameters
    ----------
    arguments : tuple
        A tuple containing:
        - package (Package): A Package object with a subset of batches.
        - (start_idx, end_idx) (tuple of int): The batch indices to process.
        - stepsize (float): The time step for the simulation.
        - timesteps (int): The number of time steps.
        - solver (str): The name of the solver to use.

    Returns
    -------
    tuple
        A tuple of the resulting state dictionaries and energy vector:
        (InventoryStates, ActivityStates, HeatStates, HeatSumStates,
        EnergyVector).
    """
    package, (start_idx, end_idx), stepsize, timesteps, solver = arguments
    # Inputs need to be initial state.
    # Calculate the Bateman Matrix.
    # Calculate new states.
    package.calculate_states(stepsize=stepsize, timesteps=timesteps, solver=solver)
    
    return package.InventoryStates, package.ActivityStates, package.HeatStates, package.HeatSumStates, package.EnergyVector

def parallel_simulate(package, stepsize, timesteps, solver, num_of_cores):
    """Manages the parallel simulation of a single package with multiple
    batches.

    This function distributes the simulation batches of a single Package object
    across multiple CPU cores for parallel processing. It divides the package's
    data into chunks, sends each chunk to a worker process, and then combines
    the results to reconstruct the complete simulation state.

    Parameters
    ----------
    package : Package
        The Package object to be simulated.
    stepsize : float
        The time step for the simulation.
    timesteps : int
        The number of time steps.
    solver : str
        The name of the solver to be used.
    num_of_cores : int, optional
        The number of cores to use. If None, it uses the lesser of the
        system's available cores or the number of batches. The default is None.

    Returns
    -------
    Package
        The original Package object, updated with the full simulation results.
    """
    
    # Make a list of modified packages that feeds into the parallel_calculate_states function.
    if num_of_cores == None:
        process_count = min(multiprocessing.cpu_count(), package.batches)    
    else:
        process_count = min(min(multiprocessing.cpu_count(), num_of_cores), package.batches)

    individual_batches = int(package.batches / process_count)
    
    while individual_batches < 1:
        process_count = int(process_count-1)
        individual_batches = int(package.batches / process_count)

    residual_batches = package.batches - process_count * individual_batches

    list_of_arguments = []
    for i in range(process_count):
        if i == 0:
            start_idx = i * individual_batches 
            end_idx = start_idx + individual_batches + residual_batches
        else:
            start_idx = i * individual_batches + residual_batches
            end_idx = start_idx + individual_batches

        TempPackage = copy.deepcopy(package)
        
        # Division into proper-sized chunks.
        if len(TempPackage.InventoryStates) != 0:
            for time in TempPackage.InventoryStates.keys():
                TempPackage.InventoryStates[time] = TempPackage.InventoryStates[time][:,start_idx:end_idx]
                TempPackage.ActivityStates[time]  = TempPackage.ActivityStates[time][:,start_idx:end_idx]
                TempPackage.HeatStates[time]      = TempPackage.HeatStates[time][:,start_idx:end_idx]
                TempPackage.HeatSumStates[time]   = TempPackage.HeatSumStates[time][start_idx:end_idx]
                
        # Prepare the package Inventories.
        TempInvValues = np.array(list(TempPackage.Inventory.values()))[:,start_idx:end_idx]
        TempInvKeys = TempPackage.Inventory.keys()
    
        TempInv = dict(zip(TempInvKeys, TempInvValues))
        TempPackage.Inventory = TempInv
        
        list_of_arguments.append([TempPackage, (start_idx, end_idx), stepsize, timesteps, solver])

    # Create a pool of worker processes
    if process_count == 1:
        results = []
        for arg in list_of_arguments:
            results.append(parallel_calculate_states(arg))
        
        TimeStates = results[0][0].keys()        
        IStates = np.array(list(results[0][0].values()))
        AStates = np.array(list(results[0][1].values()))
        HStates = np.array(list(results[0][2].values()))
        HSStates = np.array(list(results[0][3].values()))
        EnergyVector = results[0][4]
        
        for idx, (IS, AS, HS, HSS, EV) in enumerate(results):
            if idx == 0:
                continue
            
            IStates = np.concatenate((IStates, np.array(list(IS.values()))), axis=2)
            AStates = np.concatenate((AStates, np.array(list(AS.values()))), axis=2)
            HStates = np.concatenate((HStates, np.array(list(HS.values()))), axis=2)
            HSStates = np.concatenate((HSStates, np.array(list(HSS.values()))), axis=1)

    else:
        
        with multiprocessing.Pool(processes=process_count) as pool:
            # Distribute the work across the worker processes
            results = pool.map(parallel_calculate_states, list_of_arguments)
            
            TimeStates = results[0][0].keys()        
            IStates = np.array(list(results[0][0].values()))
            AStates = np.array(list(results[0][1].values()))
            HStates = np.array(list(results[0][2].values()))
            HSStates = np.array(list(results[0][3].values()))
            EnergyVector = results[0][4]
            
            for idx, (IS, AS, HS, HSS, EV) in enumerate(results):
                if idx == 0:
                    continue
                
                IStates = np.concatenate((IStates, np.array(list(IS.values()))), axis=2)
                AStates = np.concatenate((AStates, np.array(list(AS.values()))), axis=2)
                HStates = np.concatenate((HStates, np.array(list(HS.values()))), axis=2)
                HSStates = np.concatenate((HSStates, np.array(list(HSS.values()))), axis=1)
            
    # Create Proper dictionaries to finalize the state formats.
    InventoryStates = {}; ActivityStates = {}; HeatStates = {}; HeatSumStates = {}
    for idx, time in enumerate(TimeStates):
        InventoryStates[time] = IStates[idx]
        ActivityStates[time]  = AStates[idx]
        HeatStates[time]      = HStates[idx]  
        HeatSumStates[time]   = HSStates[idx]  
        
    package.InventoryStates = InventoryStates
    package.ActivityStates = ActivityStates
    package.HeatStates = HeatStates
    package.HeatSumStates = HeatSumStates
    package.EnergyVector = EnergyVector
    
    return package

def parallel_simulate_package(packages, stepsize, timesteps, solver, num_of_cores):
    """Manages the parallel simulation of a list of packages.

    This function distributes a list of individual Package objects across
    multiple CPU cores for parallel processing. It assigns each package to a
    worker process and then updates each original package with the
    simulation results.

    Parameters
    ----------
    packages : list of Package
        A list of Package objects to be simulated.
    stepsize : float
        The time step for the simulation.
    timesteps : int
        The number of time steps.
    solver : str
        The name of the solver to be used.
    num_of_cores : int, optional
        The number of cores to use. If None, it defaults to the lesser of the
        system's available cores or the number of packages. The default is None.

    Returns
    -------
    list of Package
        The original list of Package objects, updated with the simulation
        results.
    """
    
    # Make a list of modified packages that feeds into the parallel_calculate_states function.
    if num_of_cores == None:
        process_count = min(multiprocessing.cpu_count(), len(packages))    
    else:
        process_count = min(min(multiprocessing.cpu_count(), num_of_cores), len(packages))

    list_of_arguments = []
    for i in range(len(packages)):
        TempPackage = copy.deepcopy(packages[i])
                
        # Prepare the package Inventories.
        TempInvValues = np.array(list(TempPackage.Inventory.values()))
        TempInvKeys = TempPackage.Inventory.keys()
    
        TempInv = dict(zip(TempInvKeys, TempInvValues))
        TempPackage.Inventory = TempInv
        
        list_of_arguments.append([TempPackage, (0, 0), stepsize, timesteps, solver])

    # Create a pool of worker processes
    if process_count == 1:
        results = []
        for arg in list_of_arguments:
            results.append(parallel_calculate_states(arg))

            for idx, (IS, AS, HS, HSS, EV) in enumerate(results):
                TimeStates = results[idx][0].keys()        
                EnergyVector = results[idx][4]
                
                IStates = np.array(list(IS.values()))
                AStates = np.array(list(AS.values()))
                HStates = np.array(list(HS.values()))
                HSStates = np.array(list(HSS.values()))
            
                # Create Proper dictionaries to finalize the state formats.
                InventoryStates = {}; ActivityStates = {}; HeatStates = {}; HeatSumStates = {}
                for jdx, time in enumerate(TimeStates):
                    InventoryStates[time] = IStates[jdx]
                    ActivityStates[time]  = AStates[jdx]
                    HeatStates[time]      = HStates[jdx]  
                    HeatSumStates[time]   = HSStates[jdx]  
    
                packages[idx].InventoryStates = InventoryStates
                packages[idx].ActivityStates = ActivityStates
                packages[idx].HeatStates = HeatStates
                packages[idx].HeatSumStates = HeatSumStates
                packages[idx].EnergyVector = EnergyVector

    else:
        with multiprocessing.Pool(processes=process_count) as pool:
            # Distribute the work across the worker processes
            results = pool.map(parallel_calculate_states, list_of_arguments)
            
            for idx, (IS, AS, HS, HSS, EV) in enumerate(results):
                TimeStates = results[idx][0].keys()        
                EnergyVector = results[idx][4]
                
                IStates = np.array(list(IS.values()))
                AStates = np.array(list(AS.values()))
                HStates = np.array(list(HS.values()))
                HSStates = np.array(list(HSS.values()))
            
                # Create Proper dictionaries to finalize the state formats.
                InventoryStates = {}; ActivityStates = {}; HeatStates = {}; HeatSumStates = {}
                for jdx, time in enumerate(TimeStates):
                    InventoryStates[time] = IStates[jdx]
                    ActivityStates[time]  = AStates[jdx]
                    HeatStates[time]      = HStates[jdx]  
                    HeatSumStates[time]   = HSStates[jdx]  
    
                packages[idx].InventoryStates = InventoryStates
                packages[idx].ActivityStates = ActivityStates
                packages[idx].HeatStates = HeatStates
                packages[idx].HeatSumStates = HeatSumStates
                packages[idx].EnergyVector = EnergyVector
    
    return packages