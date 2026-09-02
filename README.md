# wastimate

wastimate is a Python package for simulating the time evolution of radioactive waste inventories, accounting for radioactive decay and the transfer of material between locations. Given a description of a set of waste packages, their locations, and the rules governing their movement or introduction over time, the package computes mass, volume, activity, and heat output at arbitrary points in the simulated timeline.

The package is intended for inventory and dose-relevant quantity estimation in waste management contexts: projecting activity or heat output at a future date, evaluating the effect of decay on short-lived isotopes within a package, or tracking total inventory at a facility as material is added, removed, or transferred according to a defined schedule.

## Model structure

A wastimate model is composed of a small number of entity types.

A package represents a discrete unit of radioactive material (a drum, canister, or other physical unit), defined by a nuclide inventory. Inventories may be specified directly or imported from tabular data.

A node represents a physical or logical location at which packages reside. A model typically defines multiple nodes.

An order defines a rule for transferring packages meeting specified criteria from one node to another, executed according to a defined schedule. A source defines the introduction of new packages into a node over time, for cases in which waste generation or receipt occurs during the simulated period rather than being present at initialization.

Nuclide decay is computed from decay chain data for every nuclide in every package at each simulation step. Package composition, mass, activity, and heat output are therefore correct for any simulated time, not only the initial state.

## Execution

A model is defined in a single XML file specifying the packages, nodes, orders, sources, and the sequence of actions to be executed (time advancement, result output, package transfer, and so on). A model is executed with a single function call:

    from wastimate.xmlapi import xml_simulate as xmlsim
    xmlsim.run("model.xml")

This entry point inspects the model file and dispatches execution accordingly. A fully deterministic model is executed once. If any parameter in the model is specified as a distribution rather than a fixed value, a Monte Carlo ensemble is executed automatically, without requiring a separate function call.

## Uncertainty quantification and Monte Carlo analysis

Any numeric parameter in a model, including nuclide inventory values, transfer rates, and criterion thresholds, may be specified as a probability distribution (normal, uniform, log-normal, or triangular) rather than a fixed value. When a model contains one or more distributions, execution proceeds as a Monte Carlo ensemble: each trial samples independent parameter values, trials are executed independently (optionally in parallel across multiple processor cores), and results across all trials are aggregated automatically into a single output file upon completion. Both simple random sampling and Latin Hypercube sampling are supported, the latter providing improved coverage of the parameter space for a given number of trials.

Sampled parameter values for each individual trial are recorded alongside the aggregated results, permitting inspection of the parameter set underlying any specific trial's output.

## Output

Simulation output is written as JSON, comprising mass, volume, activity, heat, and other computed variables as a function of time, disaggregated by node or package and, where applicable, by nuclide. Full state snapshots may be saved at specified points in the simulation, capturing the complete simulation state required for inspection or resumption, including a record of any packages merged during execution and their constituent contents.

## Configuration and visualization tool

This repository includes a standalone HTML application (index.html) for model construction and results visualization. The application requires no installation or external dependencies beyond a web browser.

The configuration interface provides editing of packages, nodes, orders, and sources, including a graphical network editor for node connectivity, a live preview of the generated model XML, and support for specifying any parameter as a distribution for Monte Carlo execution. The interface performs basic validation of model completeness, flagging conditions such as the absence of defined nodes or packages with no specified nuclides prior to execution.

The visualization interface reads JSON simulation output and renders it as time series and tabular data. For Monte Carlo output, the interface additionally provides per-node distribution plots, cross-node comparison with standard deviation bands, and time series views showing individual trials alongside the aggregated mean and spread. Chart output may be exported in raster (PNG) or vector (SVG) format.

The application also supports import of tabular inventory data, with an option to apply a specified uncertainty distribution to imported values for Monte Carlo model construction.
