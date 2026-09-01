# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:08:05 2025

@author: hando
"""
import xml.etree.ElementTree as ET
from wastimate.core import combinedpackage as cge
from wastimate.xmlapi import xml_convert as xmlc
import numpy as np
import json

def save_state(verse, results_file="results.xml"):
    """Serializes the current state of a Universe object from the simulation
    into a structured XML file. This is useful for saving the progress of a
    simulation or for creating a checkpoint that can be loaded later.

    Parameters
    ----------
    verse : An instance of the Universe class
        Holds the complete state of the simulation, including packages, nodes,
        and simulation history.
    results_file : str, optional
        The name of the XML file where the state will be saved. Defaults to 
        "results.xml".

    Returns
    -------
    None
    """

    root = ET.Element("wastimate_state", attrib={"version": "3.0.0"})

    ### SAVE PACKAGES ###
    PackagesElement = ET.SubElement(root, "packages")
    for bidx, basepack in enumerate(verse.BasePackageList):            
        if bidx == 0:
            # HalflifeDict
            NuclidesElement = ET.SubElement(PackagesElement, "nuclides")
            
            for nuc in list(basepack.NuclideDict.keys()):
                NuclideElement = ET.SubElement(NuclidesElement, "nuclide", attrib={"name":nuc,
                                                                                   "index":str(basepack.NuclideDict[nuc]),
                                                                                   "halflife":str(basepack.HalfLifeDict[nuc]),
                                                                                   "decay_energy":str(basepack.EnergyDict[nuc])})
                if nuc in basepack.MotherDaughterDict:
                    ProgenyElement = ET.SubElement(NuclideElement, "progeny")
                    ProgenyElement.text = " ".join(map(str, basepack.MotherDaughterDict[nuc])) + " " + " ".join(map(str, basepack.BranchingDict[nuc])) + " " + " ".join(map(str, basepack.DecayTypeDict[nuc]))
        
        PackageElement = ET.SubElement(PackagesElement, "package", attrib={"label":basepack.label})

        # InventoryStates
        KeyString = ' '.join(map(str, np.array(list(basepack.InventoryStates.keys()), dtype=float)))
        KeysElement = ET.SubElement(PackageElement, "keys")
        KeysElement.text = KeyString
        
        Values = np.stack(list(basepack.InventoryStates.values()), axis=0)
        InventoryElement = ET.SubElement(PackageElement, "inventory", attrib={"shape":Values.shape})        
        InventoryElement.text = ' '.join(map(str, Values.flatten()))
        
        # ActivityStates
        Values = np.stack(list(basepack.ActivityStates.values()), axis=0)
        ActivityElement = ET.SubElement(PackageElement, "activity", attrib={"shape":Values.shape})
        ActivityElement.text = ' '.join(map(str, Values.flatten()))
        
        # HeatStates
        Values = np.stack(list(basepack.HeatStates.values()), axis=0)
        HeatElement = ET.SubElement(PackageElement, "heat", attrib={"shape":Values.shape})        
        HeatElement.text = ' '.join(map(str, Values.flatten()))
        
        # HeatSumStates
        Values = np.stack(list(basepack.HeatSumStates.values()), axis=0)
        HeatSumElement = ET.SubElement(PackageElement, "heat_sum", attrib={"shape":Values.shape})
        HeatSumElement.text = ' '.join(map(str, Values.flatten()))
        
    ### SAVE COMBINED PACKAGE INFORMATION ###
    CombinedPackagesElement = ET.SubElement(root, "combined_packages")
    unique_combined_packages = []
    for verse_node in verse.Nodes:
        for time, combinedpackage in verse_node.PackageList:
            if isinstance(combinedpackage, cge.CombinedPackage) and combinedpackage not in unique_combined_packages:
                CombinedPackageElement = ET.SubElement(CombinedPackagesElement, "combined_package", attrib={"label":combinedpackage.label})
                for unique_package_state, factor in combinedpackage.PackageStates:
                    ComponentStatesElement = ET.SubElement(CombinedPackageElement, "component", attrib={"label":unique_package_state[1].label, "age":str(unique_package_state[0])})
                    if np.isscalar(factor):
                        ComponentStatesElement.text = str(factor)
                    else:
                        ComponentStatesElement.text = " ".join([str(f) for f in factor])
                unique_combined_packages.append(combinedpackage)
    if len(unique_combined_packages) == 0:
        root.remove(CombinedPackagesElement)

    ### SAVE NODE INFORMATION ###
    NodesElement = ET.SubElement(root, "nodes")
    for node in verse.Nodes:
        node_mass = node.get_mass()[1]
        node_volume = node.get_volume()[1]
        NodeElement = ET.SubElement(NodesElement, "node", attrib={"label":node.label,
                                                                  "mass":str(node_mass),
                                                                  "volume":str(node_volume)})
        LabelElement = ET.SubElement(NodeElement, "label")
        LabelElement.text = " ".join(map(str, [n[1].label for n in node.PackageList]))
        AgeElement = ET.SubElement(NodeElement, "age")
        AgeElement.text = " ".join(map(str, [float(float(n[0])) for n in node.PackageList]))

    ### SAVE NODEPACKAGEHISTORY ###
    UniverseElement = ET.SubElement(root, "universe")
    SimTimeElement = ET.SubElement(UniverseElement, "simulation_time")
    SimTimeElement.text = " ".join(map(str, verse.SimTimeHistory))
    HistoryIdxElement = ET.SubElement(UniverseElement, "reference")

    PackageLabels = " ".join([p.label for p in verse.HistoryIdx.keys()])
    PackageIdxs = " ".join(map(str, verse.HistoryIdx.values()))
    HistoryIdxElement.text = PackageLabels + " " + PackageIdxs
    HistoryElement = ET.SubElement(UniverseElement, "history")
    HistoryElement.text = json.dumps(verse.NodePackageHistory)
        
    # Convert to a formatted XML string
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)  # Pretty-print formatting
    tree.write(results_file, encoding="utf-8", xml_declaration=True)

def load_state(results_file):
    """ Restores a Wastimate simulation from a previously saved XML file.
    It parses the XML data and reconstructs the various dictionaries and lists
    that define the simulation's state, including the properties of packages,
    nodes, and the simulation history. This enables you to resume a simulation
    from a specific checkpoint.

    Parameters
    ----------
    results_file : str
        A string representing the file path to the XML file containing the
        saved simulation state.

    Returns
    -------
    BasePackageDict : dict
        Dictionary where each key is a package's label (a string), and the
        value is a list containing a set of properties for that package.
        This includes information such as the HalfLifeDict, BranchingDict,
        MotherDaughterDict, DecayTypeDict, EnergyDict, and NuclideDict,
        as well as time-dependent state dictionaries for Inventory, Activity,
        Heat, and HeatSum.
    CombinedPackagesDict : dict
        Dictionary holds data for combined packages. The keys are the combined
        package labels (strings). Each value is a list of lists, where each
        inner list contains information about a component package, including
        its age and a scaling factor.
    NodesDict : dict
        dictionary maps each node's label (a string) to its corresponding
        package list. The package list is a list of lists, with each inner list
        containing the age of a package and its label.
    SimTimeHistory : list
        List of floats that represents the timeline of the simulation. Each
        float corresponds to a timestamp, where a simulation step was recorded.
    HistoryIdx : dict
        Dictionary provides a reference for the simulation history. It maps
        each node's label (a string) to an integer index that helps locate
        the node's data within the NodePackageHistory.
    NodePackageHistory : dict
        Dictionary containing the historical state of all packages within each
        node throughout the simulation. It's stored as a JSON object, which 
        parsed by the LoadState function to recreate the full history.
    """

    tree = ET.parse(results_file)
    root = tree.getroot()
    
    BasePackageDict = {}
    PackagesElement = root.find("packages")
    NuclidesElement = PackagesElement.find("nuclides")
    for PackageElement in PackagesElement:   
        HalfLifeDict = {}
        EnergyDict = {}
        NuclideDict = {}
        InvNuclideDict = {}
        BranchingDict = {}
        MotherDaughterDict = {}
        DecayTypeDict = {}
        for NuclideElement in NuclidesElement:
            HalfLifeDict[NuclideElement.attrib["name"]] = float(NuclideElement.attrib["halflife"])
            EnergyDict[NuclideElement.attrib["name"]] = float(NuclideElement.attrib["decay_energy"])
            NuclideDict[NuclideElement.attrib["name"]] = int(NuclideElement.attrib["index"])
            InvNuclideDict[int(NuclideElement.attrib["index"])] = NuclideElement.attrib["name"]
            
            ProgenyElement = NuclideElement.find("progeny")
            if ProgenyElement is not None:
                parts = xmlc.split_text(ProgenyElement.text)
                midpoint = len(parts) // 3
                nucs = parts[:midpoint]
                factors = list(map(float, parts[midpoint:midpoint+midpoint]))
                decays = list(parts[midpoint+midpoint:])
                MotherDaughterDict[NuclideElement.attrib["name"]] = nucs
                BranchingDict[NuclideElement.attrib["name"]] = factors
                DecayTypeDict[NuclideElement.attrib["name"]] = decays
        
        # Keys
        KeysElement = PackageElement.find("keys")
        Keys = xmlc.split_text(KeysElement.text, cast=float)
        
        # InventoryStates
        InventoryElement = PackageElement.find("inventory")
        Shape = tuple(map(int, InventoryElement.attrib["shape"].replace("(", "").replace(")", "").split(",")))
        Values = np.array(xmlc.split_text(InventoryElement.text), dtype=float).reshape(Shape)
        InventoryStates = dict(zip(Keys, Values))
        
        # ActivityStates
        ActivityElement = PackageElement.find("activity")
        Shape = tuple(map(int, ActivityElement.attrib["shape"].replace("(", "").replace(")", "").split(",")))
        Values = np.array(xmlc.split_text(ActivityElement.text), dtype=float).reshape(Shape)
        ActivityStates = dict(zip(Keys, Values))
        
        # HeatStates
        HeatElement = PackageElement.find("heat")
        Shape = tuple(map(int, HeatElement.attrib["shape"].replace("(", "").replace(")", "").split(",")))
        Values = np.array(xmlc.split_text(HeatElement.text), dtype=float).reshape(Shape)
        HeatStates = dict(zip(Keys, Values))
        
        # HeatSumStates
        HeatSumElement = PackageElement.find("heat_sum")
        Shape = tuple(map(int, HeatSumElement.attrib["shape"].replace("(", "").replace(")", "").split(",")))
        Values = np.array(xmlc.split_text(HeatSumElement.text), dtype=float).reshape(Shape)
        HeatSumStates = dict(zip(Keys, Values))

        # Create a package
        BasePackageDict[PackageElement.attrib["label"]] = [HalfLifeDict, BranchingDict, MotherDaughterDict, DecayTypeDict, EnergyDict, NuclideDict,
                                                           InvNuclideDict, InventoryStates, ActivityStates, HeatStates, HeatSumStates]

        # Create CombinedPackagesDict
        CombinedPackagesDict = {}
        CombinedPackagesElement = root.find("combined_packages")
        if CombinedPackagesElement is not None:
            for CombinedPackageElement in CombinedPackagesElement:
                combinedpackagelabel = CombinedPackageElement.attrib["label"]
                CombinedPackageStates = []
                for ComponentStatesElement in CombinedPackageElement:
                    componentlabel = ComponentStatesElement.attrib["label"]
                    componentage = float(ComponentStatesElement.attrib["age"])
                    componentfactor = xmlc.split_text(ComponentStatesElement.text, cast=float)
                    if len(componentfactor) == 1:
                        componentfactor = componentfactor[0]

                    CombinedPackageStates.append([[componentage, componentlabel], componentfactor])
                CombinedPackagesDict[combinedpackagelabel] = CombinedPackageStates
        
        # Make a Nodes Dictionary
        NodesElement = root.find("nodes")
        NodesDict = {}
        for NodeElement in NodesElement:
            labels = xmlc.split_text(NodeElement.find("label").text)
            if labels:
                ages = xmlc.split_text(NodeElement.find("age").text, cast=float)
                PackageList = [list(pair) for pair in zip(ages, labels)]
                NodesDict[NodeElement.attrib["label"]] = PackageList
            else:
                NodesDict[NodeElement.attrib["label"]] = []
            
        ### Load History Related Activities ###
        UniverseElement = root.find("universe")
        SimTimeElement = UniverseElement.find("simulation_time")
        SimTimeHistory = xmlc.split_text(SimTimeElement.text, cast=float)
        
        HistoryIdxElement = UniverseElement.find("reference")
        parts = xmlc.split_text(HistoryIdxElement.text)
        midpoint = len(parts) // 2
        nodes_labels = parts[:midpoint]
        node_idx = list(map(int, parts[midpoint:]))
        HistoryIdx = dict(zip(nodes_labels, node_idx))        

        HistoryElement = UniverseElement.find("history")    
        NodePackageHistory = json.loads(HistoryElement.text)   
        
    return BasePackageDict, CombinedPackagesDict, NodesDict, SimTimeHistory, HistoryIdx, NodePackageHistory