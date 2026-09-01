# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:08:05 2025

@author: hando
"""

from wastimate.core import package as ge
from wastimate.core import combinedpackage as cge
from wastimate.core import node as de
from wastimate.core import source as rce
from wastimate.core import order as der
from wastimate.core import universe as uni

from wastimate.xmlapi import xml_state as xmls
from wastimate.xmlapi import xml_convert as xmlc

from tqdm import tqdm
import tqdm as _tqdm_module
import xml.etree.ElementTree as ET
import numpy as np
import json
import pandas as pd
import os
import copy
import re
import multiprocessing
import glob
from scipy.stats import qmc, norm as sp_norm, uniform as sp_uniform, lognorm as sp_lognorm, triang as sp_triang


# ===================== MONTE CARLO / DISTRIBUTIONS =====================
# Any attribute value in the model XML can be a plain number, or a distribution written as
# "type(params)" -- e.g. normal(1e10, 1e9) -- matching exactly the syntax the HTML config
# builder writes. This layer is intentionally self-contained: nothing about the rest of
# xml_simulate.py, or any of the Package/Node/Order/Source classes, needs to know
# distributions exist. A distribution string is always resolved down to a plain float --
# either its expected value (for a normal deterministic run) or a sampled value (per Monte
# Carlo trial) -- before anything else ever reads the attribute.
_DIST_PATTERN = re.compile(r'^\s*([a-zA-Z_]+)\s*\(([^)]*)\)\s*$')


def parse_distribution_string(value):
    """Returns (dist_type, params) if `value` is a distribution spec like
    "normal(1e10, 1e9)", or None if it's a plain number (or anything else).
    """
    if not isinstance(value, str):
        return None
    m = _DIST_PATTERN.match(value)
    if not m:
        return None
    dist_type = m.group(1).lower()
    if dist_type not in ("normal", "uniform", "lognormal", "triangular"):
        return None
    try:
        params = [float(p.strip()) for p in m.group(2).split(",")]
    except ValueError:
        return None
    expected_len = {"normal": 2, "uniform": 2, "lognormal": 2, "triangular": 3}[dist_type]
    if len(params) != expected_len:
        return None
    return dist_type, params


def distribution_expected_value(dist_type, params):
    """The mean of the distribution -- used to resolve a distribution down to a single
    representative number for a normal (non-Monte-Carlo) run."""
    if dist_type == "normal":
        mean, _std = params
        return mean
    if dist_type == "uniform":
        low, high = params
        return (low + high) / 2.0
    if dist_type == "lognormal":
        mu, sigma = params
        return float(np.exp(mu + sigma**2 / 2.0))
    if dist_type == "triangular":
        low, mode, high = params
        return (low + mode + high) / 3.0
    raise ValueError(f"Unknown distribution type: {dist_type}")


def sample_distribution(dist_type, params, rng):
    """Draws one independent sample from the distribution using a numpy Generator."""
    if dist_type == "normal":
        mean, std = params
        return float(rng.normal(mean, std))
    if dist_type == "uniform":
        low, high = params
        return float(rng.uniform(low, high))
    if dist_type == "lognormal":
        mu, sigma = params
        return float(rng.lognormal(mu, sigma))
    if dist_type == "triangular":
        low, mode, high = params
        return float(rng.triangular(low, mode, high))
    raise ValueError(f"Unknown distribution type: {dist_type}")


def lhs_transform_column(unit_samples, dist_type, params):
    """Transforms a column of Latin Hypercube unit-interval samples (values in [0, 1), one
    per trial) into samples from the actual target distribution, via its inverse CDF. This
    is what makes Latin Hypercube sampling work for arbitrary marginal distributions: the
    stratification happens in the [0, 1) unit cube, then each parameter's own inverse CDF
    maps that stratified design into its real sample space.
    """
    if dist_type == "normal":
        mean, std = params
        return sp_norm.ppf(unit_samples, loc=mean, scale=std)
    if dist_type == "uniform":
        low, high = params
        return sp_uniform.ppf(unit_samples, loc=low, scale=high - low)
    if dist_type == "lognormal":
        mu, sigma = params
        return sp_lognorm.ppf(unit_samples, s=sigma, scale=np.exp(mu))
    if dist_type == "triangular":
        low, mode, high = params
        c = (mode - low) / (high - low)
        return sp_triang.ppf(unit_samples, c, loc=low, scale=high - low)
    raise ValueError(f"Unknown distribution type: {dist_type}")


_INTEGER_ATTRS = {"magnitude", "rate"}


def _coerce_if_integer_attr(attr_name, value):
    """magnitude and rate (on orders and sources alike) are counts of packages or of
    simulation steps, not continuous quantities -- whenever a sampled or expected value
    lands on one of these attributes, round it to the nearest integer rather than leaving
    a fractional value that wouldn't make physical sense (e.g. "move 4.7 packages")."""
    if attr_name in _INTEGER_ATTRS:
        return int(round(value))
    return value


def _resolve_distributions_expected(root):
    """Walks the whole tree in place and replaces every distribution-string attribute with
    its expected value, so a normal (non-Monte-Carlo) run stays fully deterministic and
    every downstream class only ever sees plain numbers, exactly as before this feature.
    """
    for elem in root.iter():
        for attr_name in list(elem.attrib.keys()):
            parsed = parse_distribution_string(elem.attrib[attr_name])
            if parsed is not None:
                dist_type, params = parsed
                value = distribution_expected_value(dist_type, params)
                value = _coerce_if_integer_attr(attr_name, value)
                elem.set(attr_name, str(value))
    return root


def _find_distribution_locations(root):
    """Returns a list of (element, attrib_name, dist_type, params) for every distribution
    found anywhere in the tree, in a stable (tree-walk) order."""
    locations = []
    for elem in root.iter():
        for attr_name, attr_val in elem.attrib.items():
            parsed = parse_distribution_string(attr_val)
            if parsed is not None:
                dist_type, params = parsed
                locations.append((elem, attr_name, dist_type, params))
    return locations


def _build_sample_matrix(dist_locations, n_trials, sampling, seed):
    """Builds an (n_trials x n_distributions) matrix of sampled values, either via plain
    independent random sampling or Latin Hypercube sampling (which stratifies each
    parameter's range across the trials for better coverage with fewer trials)."""
    n_params = len(dist_locations)
    matrix = np.zeros((n_trials, n_params))
    if sampling == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        unit_samples = sampler.random(n=n_trials)  # shape (n_trials, n_params), each column in [0, 1)
        for j, (_elem, _attr, dist_type, params) in enumerate(dist_locations):
            matrix[:, j] = lhs_transform_column(unit_samples[:, j], dist_type, params)
    else:
        rng = np.random.default_rng(seed)
        for j, (_elem, _attr, dist_type, params) in enumerate(dist_locations):
            for i in range(n_trials):
                matrix[i, j] = sample_distribution(dist_type, params, rng)
    return matrix


def _describe_location(elem, attr_name, orig_elems, parent_of):
    """Best-effort human-readable label for a distributed parameter's location -- e.g.
    "package:Drum1.mass" or "nuclide:Drum1/Co60.value" -- so a trial's sampled values can
    be shown without needing to open that trial's own XML file.
    """
    own_label = elem.attrib.get("label")
    if own_label:
        return f"{elem.tag}:{own_label}.{attr_name}"
    parent = parent_of.get(id(elem))
    own_name = elem.attrib.get("name")
    if parent is not None:
        parent_label = parent.attrib.get("label") or parent.attrib.get("name")
        if parent_label and own_name:
            return f"{elem.tag}:{parent_label}/{own_name}.{attr_name}"
        if parent_label:
            return f"{parent.tag}:{parent_label}>{elem.tag}.{attr_name}"
    if own_name:
        return f"{elem.tag}:{own_name}.{attr_name}"
    return f"{elem.tag}[{orig_elems.index(elem)}].{attr_name}"


def _build_parent_map(root):
    parent_of = {}
    for parent in root.iter():
        for child in parent:
            parent_of[id(child)] = parent
    return parent_of


def _build_trial_xml(root, dist_locations, location_labels, sample_row, trial_idx, output_dir):
    """Deep-copies the template tree, substitutes this trial's sampled values in place of
    each distribution, and redirects any save_results/save_state action to a trial-specific
    file inside the results folder so trials never overwrite each other. Returns the
    resulting model as an XML string (not a live Element) so it can be handed to a
    multiprocessing worker without any pickling concerns.

    Also writes a trial_{idx}_params.json file recording every sampled value under its
    human-readable label, so a trial's exact configuration can be looked up later (e.g. by
    the HTML viewer, when hovering a trial's line) without needing the full trial XML.
    """
    trial_root = copy.deepcopy(root)
    # copy.deepcopy doesn't preserve element identity, so re-walk both trees in the same
    # (stable) order to map each original distribution location to its counterpart in the copy.
    orig_elems = list(root.iter())
    trial_elems = list(trial_root.iter())
    elem_index = {id(e): k for k, e in enumerate(orig_elems)}

    trial_params = {}
    for j, (elem, attr_name, _dist_type, _params) in enumerate(dist_locations):
        value = _coerce_if_integer_attr(attr_name, sample_row[j])
        trial_elems[elem_index[id(elem)]].set(attr_name, str(value))
        trial_params[location_labels[j]] = value

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"trial_{trial_idx:05d}_params.json"), "w") as f:
        json.dump(trial_params, f)

    universe_el = trial_root.find("universe")
    if universe_el is not None:
        for action in universe_el.findall("action"):
            for key in ("save_results", "save_state"):
                if key in action.attrib:
                    base_name = action.attrib[key]
                    action.attrib[key] = os.path.join(output_dir, f"trial_{trial_idx:05d}_{base_name}")

    return ET.tostring(trial_root, encoding="unicode")


def _silence_tqdm():
    """Monkey-patches the tqdm CLASS's __init__ itself (not a per-module name), so this
    silences every nested progress bar regardless of which module created it or how it
    imported tqdm ("import tqdm" vs "from tqdm import tqdm" both end up pointing at the
    exact same class object). Returns the original __init__ so it can be restored.
    """
    original_init = _tqdm_module.tqdm.__init__

    def _silent_init(self, *args, **kwargs):
        kwargs['disable'] = True
        original_init(self, *args, **kwargs)

    _tqdm_module.tqdm.__init__ = _silent_init
    return original_init


def _unsilence_tqdm(original_init):
    _tqdm_module.tqdm.__init__ = original_init


def _run_trial_worker(args):
    """Runs one Monte Carlo trial. Must be a plain module-level function (not a closure) so
    it can be pickled and sent to a multiprocessing worker process.

    Without silencing, every trial spawns its own "Initializing Packages" bar plus whatever
    progress bar(s) the simulation itself uses internally -- for anything beyond a handful
    of trials that floods the console with bar renders. Only the outer "Monte Carlo trials"
    bar in run_monte_carlo() should be visible; this suppresses everything nested inside a
    single trial's run_simulation() call.
    """
    trial_idx, xml_string = args
    trial_root = ET.fromstring(xml_string)
    original_init = _silence_tqdm()
    try:
        run_simulation(root=trial_root)
    finally:
        _unsilence_tqdm(original_init)
    return trial_idx


def run(model_file, **kwargs):
    """The single entry point to use for running a model -- inspects the file once and runs
    it as a normal single simulation, or as a full Monte Carlo ensemble, depending on
    whether the file actually contains any distributions anywhere (a package's nuclide
    value, an order's magnitude, a criterion threshold, etc. written as "type(params)"
    instead of a plain number). You shouldn't normally need to call run_simulation or
    run_monte_carlo directly -- this reads the file and picks the right one for you.

    Any extra keyword arguments are passed straight through to whichever of the two
    actually ends up running (e.g. n_trials/sampling/seed/output_dir/num_of_cores for a
    Monte Carlo run, or results_file for a normal one) -- see run_simulation and
    run_monte_carlo for what each accepts.

    Parameters
    ----------
    model_file : str
        Path to the XML model file.

    Returns
    -------
    None
    """
    tree = ET.parse(model_file)
    root = tree.getroot()
    if _find_distribution_locations(root):
        return run_monte_carlo(model_file, **kwargs)
    return run_simulation(model_file=model_file, **kwargs)


def run_monte_carlo(model_file, n_trials=None, sampling=None, seed=None, output_dir=None,
                     num_of_cores=None, keep_trial_files=None, aggregated_file=None):
    """Runs the model many times, each time sampling a fresh value for every distribution
    found in the file, and saves each trial's results as its own file inside `output_dir`
    (via whatever save_results/save_state actions the model already defines). Once every
    trial has run, this automatically aggregates them into one combined file (see
    aggregate_monte_carlo) -- both the snapshot results and the full time series, whichever
    the model actually produced, plus each trial's own sampled parameter values -- so
    aggregation isn't a separate step you have to remember to run afterward, and the viewer
    only has to load one file to see everything, including what each individual trial was
    actually run with.

    The aggregated file is written next to the model XML itself, not buried inside the
    (temporary, by default) per-trial output folder -- that's the one file you actually
    want to keep around. The per-trial files that fed into it are deleted automatically
    once aggregation succeeds, unless keep_trial_files is set.

    Trial count, sampling method, seed, and whether to keep the per-trial files can all be
    read straight from the model's <universe> element (mc_trials, mc_sampling, mc_seed,
    mc_keep_files attributes, as written by the HTML config builder) if not passed
    explicitly here.

    Parameters
    ----------
    model_file : str
        Path to the XML model file. Any attribute anywhere in the file may hold a
        distribution spec like "normal(1e10, 1e9)" instead of a plain number.
    n_trials : int, optional
        Number of Monte Carlo trials to run. Falls back to the model's mc_trials
        attribute, or 100 if neither is given.
    sampling : str, optional
        "latin_hypercube" or "random". Falls back to the model's mc_sampling attribute,
        or "random" if neither is given.
    seed : int, optional
        Random seed for reproducibility. Falls back to the model's mc_seed attribute.
    output_dir : str, optional
        Folder each trial's results/state files are written into while the run is in
        progress. Defaults to a "results" folder next to the model file. Deleted trial by
        trial as aggregation consumes them, unless keep_trial_files is set.
    num_of_cores : int, optional
        Number of worker processes to run trials in parallel with. Falls back to the
        model's universe num_of_cores attribute, or runs serially if that's 1 or unset.
    keep_trial_files : bool, optional
        If True, per-trial files are left in output_dir instead of being deleted after
        aggregation. Falls back to the model's mc_keep_files attribute, or False.
    aggregated_file : str, optional
        Where to write the combined aggregated .json. Defaults to a file named after the
        model (e.g. "my_model.xml" -> "my_model_mc_aggregated.json") in the same folder as
        the model file itself.

    Returns
    -------
    None
    """
    model_path = os.path.abspath(model_file)
    model_dir = os.path.dirname(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0]

    tree = ET.parse(model_file)
    root = tree.getroot()
    universe_el = root.find("universe")

    if n_trials is None:
        n_trials = int(universe_el.attrib.get("mc_trials", 100)) if universe_el is not None else 100
    if sampling is None:
        sampling = universe_el.attrib.get("mc_sampling", "random") if universe_el is not None else "random"
    if seed is None and universe_el is not None and "mc_seed" in universe_el.attrib:
        seed = int(universe_el.attrib["mc_seed"])
    if num_of_cores is None:
        num_of_cores = int(universe_el.attrib.get("num_of_cores", 1)) if universe_el is not None else 1
    if keep_trial_files is None:
        keep_trial_files = (universe_el.attrib.get("mc_keep_files", "false").lower() == "true") if universe_el is not None else False
    if output_dir is None:
        output_dir = os.path.join(model_dir, "results")
    if aggregated_file is None:
        aggregated_file = os.path.join(model_dir, f"{model_stem}_mc_aggregated.json")

    dist_locations = _find_distribution_locations(root)
    if not dist_locations:
        print("Warning: no distributions found in this model -- every trial would be identical. "
              "Running a single normal simulation instead.")
        run_simulation(model_file=model_file)
        return

    os.makedirs(output_dir, exist_ok=True)
    sample_matrix = _build_sample_matrix(dist_locations, n_trials, sampling, seed)
    parent_of = _build_parent_map(root)
    orig_elems = list(root.iter())
    location_labels = [_describe_location(elem, attr_name, orig_elems, parent_of)
                        for elem, attr_name, _dt, _p in dist_locations]

    trial_args = [
        (i, _build_trial_xml(root, dist_locations, location_labels, sample_matrix[i, :], i, output_dir))
        for i in range(n_trials)
    ]

    if num_of_cores and num_of_cores > 1:
        with multiprocessing.Pool(num_of_cores) as pool:
            for _ in tqdm(pool.imap_unordered(_run_trial_worker, trial_args), total=n_trials,
                          bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
                          desc="Monte Carlo trials", colour='blue'):
                pass
    else:
        for args in tqdm(trial_args, bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
                          desc="Monte Carlo trials", colour='blue'):
            _run_trial_worker(args)

    # Aggregation is part of running a Monte Carlo ensemble, not a separate step you have to
    # remember to run afterward -- combined into one file, written next to the model itself,
    # so the viewer only needs one load to get every view regardless of whether the model
    # used save_state, save_results, or both.
    aggregated_path = aggregate_monte_carlo(output_dir=output_dir, output_file=aggregated_file)
    if aggregated_path:
        print(f"Aggregated results written to {aggregated_path}")

    if not keep_trial_files:
        for f in glob.glob(os.path.join(output_dir, "trial_*")):
            os.remove(f)
        # Only remove the folder itself if aggregation actually ran and it's now empty --
        # if aggregation found nothing (e.g. neither save_state nor save_results was used),
        # leave whatever's there rather than silently deleting something unexpected.
        if aggregated_path and not os.listdir(output_dir):
            os.rmdir(output_dir)


def _compute_snapshot_records(output_dir):
    """Returns the list of {trial, node, package_count, mass, volume} records from every
    trial_*.xml save_state file in output_dir, or None if there are none."""
    state_files = sorted(glob.glob(os.path.join(output_dir, "trial_*.xml")))
    if not state_files:
        return None

    rows = []
    for state_file in state_files:
        m = re.search(r"trial_(\d+)_", os.path.basename(state_file))
        trial_idx = int(m.group(1)) if m else -1

        tree = ET.parse(state_file)
        root = tree.getroot()
        nodes_el = root.find("nodes")
        if nodes_el is None:
            continue

        for node_el in nodes_el:
            label_el = node_el.find("label")
            package_count = len((label_el.text or "").split()) if label_el is not None else 0
            rows.append({
                "trial": trial_idx,
                "node": node_el.attrib.get("label", ""),
                "package_count": package_count,
                "mass": float(node_el.attrib.get("mass", 0.0)),
                "volume": float(node_el.attrib.get("volume", 0.0)),
            })
    return rows


def aggregate_monte_carlo_results(output_dir="results", output_file=None):
    """Aggregates the per-trial save_state snapshots produced by run_monte_carlo into a
    single .json file describing, for every node, the distribution of package count, mass,
    and volume across all trials.

    Each trial's save_state file already records every node's total mass and volume, plus
    the list of package labels present there (whose count is the package count) -- so this
    just reads that back out of every trial_*.xml file in `output_dir` and stacks it into
    one long list of records: one entry per (trial, node) pair. That shape is what the HTML
    results viewer expects, and it's also the natural shape for further analysis (e.g. load
    it and build a pandas DataFrame with pd.DataFrame(data["records"])) if you want to do
    more with it than the viewer shows.

    Most of the time you don't need to call this directly -- run_monte_carlo calls it (and
    aggregate_monte_carlo_timeseries) automatically, combined into one file via
    aggregate_monte_carlo. This standalone version is here for when you only want the
    snapshot half on its own.

    Parameters
    ----------
    output_dir : str, optional
        The folder run_monte_carlo saved its per-trial results into. Default "results".
    output_file : str, optional
        Where to write the aggregated .json. Defaults to "<output_dir>/mc_aggregated.json".

    Returns
    -------
    str
        The path the aggregated file was written to.
    """
    rows = _compute_snapshot_records(output_dir)
    if rows is None:
        raise FileNotFoundError(
            f"No trial state files found in '{output_dir}' (expected files named like "
            "trial_00001_<name>.xml). Make sure the model has a save_state action and "
            "run_monte_carlo has already been run."
        )

    if output_file is None:
        output_file = os.path.join(output_dir, "mc_aggregated.json")
    with open(output_file, "w") as f:
        json.dump({"records": rows}, f)
    return output_file


def _compute_timeseries_records(output_dir):
    """Returns the list of {trial, variable, series, nuclide, time, value} records from
    every trial_*.json save_results file in output_dir, or None if there are none."""
    result_files = sorted(f for f in glob.glob(os.path.join(output_dir, "trial_*.json")) if not f.endswith("_params.json"))
    if not result_files:
        return None

    rows = []
    for result_file in tqdm(result_files, bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
                             desc="Aggregating time series", colour='green'):
        m = re.search(r"trial_(\d+)_", os.path.basename(result_file))
        trial_idx = int(m.group(1)) if m else -1

        with open(result_file) as f:
            trial_data = json.load(f)
        times = trial_data.get("time", [])

        for variable_name, var_data in trial_data.get("variables", {}).items():
            kind = var_data.get("kind")
            series_dict = var_data.get("series", {})
            if kind == "per_nuclide":
                for series_name, nuclide_dict in series_dict.items():
                    for nuclide_name, values in nuclide_dict.items():
                        for t, v in zip(times, values):
                            rows.append({
                                "trial": trial_idx, "variable": variable_name, "series": series_name,
                                "nuclide": nuclide_name, "time": float(t), "value": float(v),
                            })
            else:
                for series_name, values in series_dict.items():
                    for t, v in zip(times, values):
                        rows.append({
                            "trial": trial_idx, "variable": variable_name, "series": series_name,
                            "nuclide": None, "time": float(t), "value": float(v),
                        })
    return rows


def aggregate_monte_carlo_timeseries(output_dir="results", output_file=None):
    """Aggregates the per-trial save_results outputs into a single long-format table
    describing every (trial, variable, series, nuclide, time) value across the whole
    ensemble -- the full time series, not just a final snapshot -- so trials can be
    compared as they evolve, not just at the end.

    Reads every trial_*.json file in `output_dir` (i.e. whatever save_results actions
    produced per trial, in the {"time": [...], "variables": {name: {"kind": ..., "series":
    ...}}} shape run_simulation now writes). "Simple" variables (mass, volume) become one
    record per (trial, series, time); "per_nuclide" variables (activity, heat, etc.) become
    one record per (trial, series, nuclide, time), with an explicit nuclide field rather
    than folding it into the series name -- there's no ambiguity to reconstruct on the way
    back out, unlike the old Excel layout this replaces.

    Most of the time you don't need to call this directly -- run_monte_carlo calls it (and
    aggregate_monte_carlo_results) automatically, combined into one file via
    aggregate_monte_carlo. This standalone version is here for when you only want the time
    series half on its own.

    Parameters
    ----------
    output_dir : str, optional
        The folder run_monte_carlo saved its per-trial results into. Default "results".
    output_file : str, optional
        Where to write the aggregated .json. Defaults to "<output_dir>/mc_timeseries.json".

    Returns
    -------
    str
        The path the aggregated file was written to.
    """
    rows = _compute_timeseries_records(output_dir)
    if rows is None:
        raise FileNotFoundError(
            f"No trial results files found in '{output_dir}' (expected files named like "
            "trial_00001_<name>.json). Make sure the model has a save_results action and "
            "run_monte_carlo has already been run."
        )

    if output_file is None:
        output_file = os.path.join(output_dir, "mc_timeseries.json")
    with open(output_file, "w") as f:
        json.dump({"records": rows}, f)
    return output_file


def _compute_trial_parameters(output_dir):
    """Returns the list of {trial, params: {label: value}} records from every
    trial_*_params.json file in output_dir, or None if there are none."""
    param_files = sorted(glob.glob(os.path.join(output_dir, "trial_*_params.json")))
    if not param_files:
        return None
    records = []
    for param_file in param_files:
        m = re.search(r"trial_(\d+)_", os.path.basename(param_file))
        trial_idx = int(m.group(1)) if m else -1
        with open(param_file) as f:
            params = json.load(f)
        records.append({"trial": trial_idx, "params": params})
    return records


def aggregate_monte_carlo(output_dir="results", output_file=None):
    """Aggregates everything a Monte Carlo run produced -- the save_state snapshots, the
    save_results time series, and each trial's own sampled parameter values -- into a
    single .json file, so the HTML viewer (or anything else consuming this) only has to
    load one file to get every view: per-node distributions, cross-node comparison, full
    time series, and "what was trial N actually run with" alike.

    This is what run_monte_carlo calls automatically once all trials finish. Whichever
    pieces a given model actually produced (a model might use only save_state, only
    save_results, or both, and may or may not have had any distributed parameters at all)
    are included; anything missing is simply left out rather than treated as an error. If
    a model produced nothing at all, this returns None instead of writing an empty file.

    Parameters
    ----------
    output_dir : str, optional
        The folder run_monte_carlo saved its per-trial results into. Default "results".
    output_file : str, optional
        Where to write the combined .json. Defaults to "<output_dir>/mc_aggregated.json".

    Returns
    -------
    str or None
        The path the aggregated file was written to, or None if there was nothing to
        aggregate (no save_state or save_results output found in output_dir).
    """
    snapshot_records = _compute_snapshot_records(output_dir)
    timeseries_records = _compute_timeseries_records(output_dir)
    parameter_records = _compute_trial_parameters(output_dir)

    if snapshot_records is None and timeseries_records is None and parameter_records is None:
        return None

    combined = {}
    if snapshot_records is not None:
        combined["snapshot"] = {"records": snapshot_records}
    if timeseries_records is not None:
        combined["timeseries"] = {"records": timeseries_records}
    if parameter_records is not None:
        combined["parameters"] = {"records": parameter_records}

    if output_file is None:
        output_file = os.path.join(output_dir, "mc_aggregated.json")
    with open(output_file, "w") as f:
        json.dump(combined, f)
    return output_file


def run_simulation(model_file=None, root=None, results_file=None):
    """The main entry point for a Wastimate simulation. It's designed to read a
    simulation model from an XML file, set up all the necessary components
    (like packages, nodes, and orders), run the simulation based on a sequence
    of instructions, and then save or plot the results.    

    The function operates by parsing the XML input file step-by-step to build
    and execute a simulation.

    1. Parsing the Model File
    The function starts by parsing the XML file to get the root element. It
    then goes through distinct sections of the XML to define the simulation
    components:

        Packages: Reads the <packages> element. Packages can be defined 
        directly within the XML or imported from external .json, .csv, or .xlsx
        files. The function handles the various data formats and uses a 
        progress bar (tqdm) for large data imports.
    
        Nodes: Reads the <nodes> element to define the various storage or
        processing locations for the packages. It handles both single packages
        and groups of packages assigned to each node.
    
        Sources & Orders: Defines the <sources> (for adding new packages)
        and <orders> (for moving or sorting packages) of the simulation. For
        orders, it also parses complex rules, including criteria based on
        nuclide concentrations, limits from external files, and instructions
        for sorting.
    
        Universe: Sets up the simulation universe, which includes the lists of
        nodes, orders, and sources, as well as the simulation time step.

    2. Executing Simulation Actions
    After setting up the simulation, the function iterates through the actions
    defined within the <universe> tag of the XML file. It uses a series of elif
    statements to interpret each action and call the appropriate simulation
    method. These actions include:

        progress_time: Advances the simulation by a specified number of time
        steps.
    
        change_stepsize: Modifies the simulation's time step.
    
        plot_results: Generates plots based on specified simulation variables
        and nodes.
    
        add_order / remove_order: Dynamically adds or removes an order from
        the active simulation.
    
        save_state / load_state: Saves or loads the entire state of the
        simulation to or from a file, allowing for checkpoints.
    
        organize_node / organize_packages: Executes sorting and optimization
        algorithms on packages within a specific node or on a set of packages.
        This involves parsing complex criteria and optimization parameters
        from the XML.
    
        save_results: Gathers simulation results (such as mass, activity,
        or heat) and saves them to a file, converting the data into a more
        usable format like a Pandas DataFrame.

    Parameters
    ----------
    model_file : str
        String representing the file path to the XML file that defines the
        simulation model. This file contains all the initial configurations for
        packages, nodes, and simulation instructions.
    results_file : str, optional
        optional string representing a file path for a results file. This 
        parameter is currently set to None by default and is intended for a
        future use case and is not implemented yet.

    Returns
    -------
    None
    """

    if root is None:
        tree = ET.parse(model_file)
        root = tree.getroot()
    _resolve_distributions_expected(root)

    # Get the universe for the data.
    UniverseElement = root.find("universe")

    # Define Packages
    ext_file_name = ""
    PackagesDict = {}
    GroupPackagesDict = {}
    total_silt_array = []
    PackagesElement = root.find("packages")
    if PackagesElement is not None:
        for package in tqdm(PackagesElement, total=len(PackagesElement), bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', desc="Initializing Packages", colour='cyan'):
            PackageDict = xmlc.convert_attrib(package.attrib)

            # secular_equilibrium can be set per-package (a "secular_equilibrium" attribute on
            # the <package> element itself -- either "all", or a space-separated list of
            # nuclide labels) -- or, for backward compatibility with older files, as a single
            # <secular_equilibrium> element applying to every package.
            if "secular_equilibrium" in PackageDict:
                if PackageDict["secular_equilibrium"] != "all":
                    PackageDict["secular_equilibrium"] = xmlc.split_text(PackageDict["secular_equilibrium"])
            else:
                EquilibriumElement = root.find("secular_equilibrium")
                if EquilibriumElement is not None:
                    PackageDict["secular_equilibrium"] = xmlc.split_text(EquilibriumElement.text)
            
            if "external_file" in PackageDict:
                if PackageDict["external_file"].split(".")[-1] == "json":
                    with open(PackageDict["external_file"], "r") as file:
                        ext_file_name = PackageDict["external_file"]
                        packagedictionary = json.load(file)
                        PackageDict.pop("external_file")
                        if "group_label" in PackageDict:
                            GroupPackagesDict[PackageDict["group_label"]] = packagedictionary["labels"]
                            PackageDict.pop("group_label")
                        total_length = len(packagedictionary["labels"])
                        for i, label in tqdm(enumerate(packagedictionary["labels"]), total=total_length, bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', desc=f"Reading {ext_file_name}", colour='MAGENTA'):   
                            NewPackageDict = PackageDict.copy()
                            if "empty_packages" in NewPackageDict:
                                NewPackageDict.pop("empty_packages")
                            NewPackageDict["batches"] = 1 #int(UniverseElement.attrib["batches"])
                            temp_dict = {}
                            for key, value in packagedictionary.items():
                                if key == "labels":
                                    NewPackageDict["label"] = value[i]
                                elif key == "mass":
                                    NewPackageDict["mass"] = float(value[i])
                                elif key == "volume":
                                    NewPackageDict["volume"] = float(value[i])
                                else:
                                    temp_dict[key] = value[i] # From Bq/g to Bq
                            
                            if "empty_packages" not in PackageDict or (not PackageDict["empty_packages"] and not all(value == 0 for value in temp_dict.values())):
                                NewPackageDict["inventory"] = temp_dict
                                PackagesDict[label] = ge.Package(**NewPackageDict)

                elif PackageDict["external_file"].split(".")[-1] in ["xls", "xlsx", "csv"]:
                    # Read all sheets into a dict
                    ext_file_name = PackageDict["external_file"]
                    all_sheets = pd.read_excel(ext_file_name, sheet_name=None, header=None)
                    PackageDict = xmlc.convert_attrib(package.attrib)

                    # Loop over each sheet
                    for sheet_name, df_raw in all_sheets.items():
                        # Column name to search for
                        col_name = "Radionukliidide kirjeldus"
                        
                        # Find all column indices where the first row equals col_name or is NaN
                        first_row = df_raw.iloc[0]
                        matching_indices = [i for i, val in enumerate(first_row) if val == col_name or pd.isna(val)]
                        
                        # Collect arrays corresponding to those columns (excluding the first row)
                        # Replace NaN with 0
                        arrays_dict = {i: df_raw.iloc[1:, i].fillna(0).tolist() for i in matching_indices}

                        # Dates
                        col_name = "Kuupäev"
                        matching_index = [i for i, val in enumerate(first_row) if val == col_name][0]
                        dates_array = (pd.to_datetime(df_raw.iloc[1:, matching_index], errors="coerce").ffill().tolist())
                        
                        # Mass
                        col_name = "Mass,"
                        matching_index = [i for i, val in enumerate(first_row) if str(val).split("\n")[0] == col_name][0]
                        mass_array = pd.to_numeric(df_raw.iloc[1:, matching_index], errors='coerce').ffill().tolist()
                        mass_unit = first_row[matching_index].split("\n")[1]
                        mass_factor_1 = (mass_unit == "g")*1 + (mass_unit == "kg")*1e3 + (mass_unit == "t")*1e6
                        
                        
                        # Maht
                        col_name = "Maht,\nm3"
                        matching_index = [i for i, val in enumerate(first_row) if val == col_name][0]
                        maht_array = pd.to_numeric(df_raw.iloc[1:, matching_index], errors='coerce').ffill().tolist()
          
                        # Tüüp
                        col_name = "Sisendi liik"
                        matching_index = [i for i, val in enumerate(first_row) if val == col_name][0]
                        sisend_array = df_raw.iloc[1:, matching_index].ffill().tolist()
                        
                        # Nimi
                        col_name = "Konteineri nr"
                        matching_index = [i for i, val in enumerate(first_row) if val == col_name][0]
                        silt_array = df_raw.iloc[1:, matching_index].ffill().tolist() 
                        total_silt_array += df_raw.iloc[2:, matching_index].ffill().tolist() 


                        if "external_file" in PackageDict:
                            PackageDict.pop("external_file")
                        if "group_label" in PackageDict:
                            GroupPackagesDict[PackageDict["group_label"]] = total_silt_array
                            PackageDict.pop("group_label")

                        total_length = len(silt_array[1:])
                        for i, label in tqdm(enumerate(silt_array[1:]), total=total_length, bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', desc=f"Reading {ext_file_name}", colour='MAGENTA'):   
                            NewPackageDict = PackageDict.copy()
                            if "empty_packages" in NewPackageDict:
                                NewPackageDict.pop("empty_packages")
                            NewPackageDict["batches"] = 1 # int(UniverseElement.attrib["batches"])
                            
                            mass_factor_2 = (sisend_array[i+1] in ["Ag"]) * 1 + (sisend_array[i+1] == "Akg") * 1e3 + (sisend_array[i+1] == "At") * 1e6
                            tempinventory = {}
                            for nuc_arr in arrays_dict.values():
                                nuc = nuc_arr[0].replace("-","")
                                val = nuc_arr[i+1]
                                if sisend_array[i+1] != "A":
                                    tempinventory[nuc] = val * mass_factor_1 / mass_factor_2
                                else:
                                    tempinventory[nuc] = val
                            
                            if "date" in PackageDict:
                                ref_date = pd.to_datetime(PackageDict["date"])
                                diff = (ref_date - dates_array[i+1]).total_seconds()
                                NewPackageDict["date"] = diff


                            NewPackageDict.update({"label":silt_array[i+1],
                                                "inventory":tempinventory,
                                                "mass":mass_array[i+1],
                                                "volume":maht_array[i+1],
                                                "mode":(((sisend_array[i+1]=="A")*"activity"+(sisend_array[i+1] in ["Ag", "Akg", "At"])*"activity_concentration"))})
                            
                            
                            if "empty_packages" not in PackageDict or (not PackageDict["empty_packages"] and not all(value == 0 for value in tempinventory.values())):
                                NewPackageDict["inventory"] = tempinventory
                                PackagesDict[label] = ge.Package(**NewPackageDict)

            else:
                # Compile the dictionary.
                inventory_dict = {}
                for nuclide in package:
                    inventory_dict[nuclide.attrib["name"]] = float(nuclide.attrib["value"])
                PackageDict["inventory"] = inventory_dict
                PackageDict["batches"] = 1 # int(UniverseElement.attrib["batches"])
                PackagesDict[package.attrib["label"]] = ge.Package(**PackageDict)
    
    # Define Nodes
    NodesDict = {}
    NodesElement = root.find("nodes")
    if NodesElement is not None:
        for node in NodesElement:
            NodeDict = xmlc.convert_attrib(node.attrib)
            packagelist = []
            if node.text is not None:
                for package_tag in xmlc.split_text(node.text):
                    val = 1
                    if "*" in package_tag:
                        package_tag, val = package_tag.split("*")
                        package_tag = package_tag.replace(" ", "")
                        val = int(val.replace(" ", ""))

                    if package_tag in PackagesDict:
                        for i in range(val):
                            if package_tag in PackagesDict:
                                packagelist.append(PackagesDict[package_tag])
                        
                    elif package_tag in GroupPackagesDict:
                        for sub_tag in GroupPackagesDict[package_tag]:
                            for i in range(val):    
                                if sub_tag in PackagesDict:
                                    packagelist.append(PackagesDict[sub_tag])

            NodeDict["packagelist"] = packagelist
            NodesDict[node.attrib["label"]] = de.Node(**NodeDict)
        
    # Define Sources
    SourcesDict = {}
    SourcesElement = root.find("sources")
    if SourcesElement is not None:
        for source in SourcesElement:
            SourceDict = xmlc.convert_attrib(source.attrib)
            SourceDict["awaynode"] = NodesDict[SourceDict["awaynode"]]
            SourceDict["package"] = PackagesDict[SourceDict["package"]]
            SourcesDict[source.attrib["label"]] = rce.Source(**SourceDict)
    
    # Define Orders
    OrdersDict = {}
    OrdersElement = root.find("orders")
    if OrdersElement is not None:
        for order in OrdersElement:
            OrderDict = xmlc.convert_attrib(order.attrib)
        
            # Handle the nodes for orders somehow.
            OrderDict["homenode"] = NodesDict[OrderDict["homenode"]]
            if "remaindernode" in OrderDict:
                OrderDict["remaindernode"] = NodesDict[OrderDict["remaindernode"]]
            if "ordernodes" in OrderDict:
                OrderDict["ordernodes"] = [NodesDict[OrderDict["ordernodes"]]]
            else:
                ordernodes = order.find("ordernodes")
                ordernodelist = [NodesDict[odr] for odr in xmlc.split_text(ordernodes.text)]
                OrderDict["ordernodes"] = ordernodelist
        
            # Deal with criteria
            ordercriteria = order.find("criteria")
            if ordercriteria is not None:
                ordercriterialist = []
                for ordercriteria in order.findall("criteria"):
                    TempCriteriaDict = xmlc.convert_attrib(ordercriteria.attrib)
                    
                    # A "node"-region criterion normally evaluates against whichever candidate
                    # source node the order is currently considering. An explicit node="Label"
                    # attribute lets it instead target any specific node in the model -- resolve
                    # that label to the actual Node object here, once, so order.py's
                    # check_criterion just uses Criteria["node"] directly without needing any
                    # lookup mechanism of its own.
                    if "node" in TempCriteriaDict:
                        TempCriteriaDict["node"] = NodesDict[TempCriteriaDict["node"]]
                    
                    if "limits" in TempCriteriaDict:
                        with open(TempCriteriaDict["limits"], "r") as file:
                            limits_dict = json.load(file)
                            TempCriteriaDict["limits"] = limits_dict
                            
                    criterianuclide = ordercriteria.find("nuclide")
                    if criterianuclide is not None:
                        crit = []
                        constraint = []
                        for crit_nuc in ordercriteria:
                            crit.append(crit_nuc.attrib["name"])
                            if "criteria" in crit_nuc.attrib:
                                constraint.append(float(crit_nuc.attrib["criteria"]))
                        TempCriteriaDict["nuclide"] = crit
                        if len(constraint) != 0:
                            TempCriteriaDict["criteria"] = constraint
                    
                    ordercriterialist.append(TempCriteriaDict)
                    
                OrderDict["criteria"] = ordercriterialist
            
            # Deal with extra conditions.
            extraconditions = order.find("conditions")
            if extraconditions is not None:
                extra_criteria = []
                for condition in extraconditions:
                    extra_criteria.append([x.strip().lower() == "true" for x in xmlc.split_text(condition.text)])
                OrderDict["extra_criteria"] = extra_criteria
            
            # Deal with instructions
            orderinstruction = order.find("instruction")
            if orderinstruction is not None:
                InstructionDict = xmlc.convert_attrib(orderinstruction.attrib)
                if "sort_node" in InstructionDict:
                    InstructionDict["sort_node"] = NodesDict[InstructionDict["sort_node"]]
                OrderDict["instruct"] = InstructionDict
    
            OrdersDict[order.attrib["label"]] = der.Order(**OrderDict)
            
    # Define the universe
    universe_labels = xmlc.split_text(UniverseElement.text)
    
    if NodesDict == {}:
        NodesList = None
    else:
        if universe_labels:
            NodesList = [NodesDict[key] for key in NodesDict.keys() if key in universe_labels]
        else:
            NodesList = list(NodesDict.values())
    
    if OrdersDict == {}:
        OrdersList = None
    else:
        if universe_labels:
            OrdersList = [OrdersDict[key] for key in OrdersDict.keys() if key in universe_labels]
        else:
            OrdersList = list(OrdersDict.values())
    
    if SourcesDict == {}:
        SourcesList = None
    else:
        if universe_labels:
            SourcesList = [SourcesDict[key] for key in SourcesDict.keys() if key in universe_labels]
        else:
            SourcesList = list(SourcesDict.values())
            
    if "step_unit" in UniverseElement.attrib:
        [stepsize] = xmlc.convert_time([float(UniverseElement.attrib["step_size"])], UniverseElement.attrib["step_unit"], mode="multiply") 
    else:
        stepsize = float(UniverseElement.attrib["step_size"])*31_536_000

    Verse = uni.Universe(nodes=NodesList,
                     orders=OrdersList,
                     sources=SourcesList,
                     stepsize=stepsize)
    
    for action in UniverseElement:
        if "progress_time" in action.attrib:
            Verse.simulate(timesteps=int(action.attrib["progress_time"]), num_of_cores=int(UniverseElement.attrib["num_of_cores"]))
        elif "change_stepsize" in action.attrib:
            Verse.stepsize = float(action.attrib["change_stepsize"])
        elif "plot_results" in action.attrib:
            labels = action.find("labels")
            if labels is not None:
                action.attrib["labels"] = xmlc.split_text(labels.text)
            else:
                action.attrib["labels"] = None
            if "nuclide" not in action.attrib:
                action.attrib["nuclide"] = None
            value = action.attrib.pop("plot_results")
            action.attrib = {"node": NodesDict[value], **action.attrib}
            
            if "filename" not in action.attrib:
                var = action.attrib["variable"]
                if "nuclide" in action.attrib:
                    nuc = action.attrib["nuclide"]
                else:
                    nuc = ""
                action.attrib["filename"] = f"{value}_{var}_{nuc}"

            Verse.plot(**action.attrib)

        elif "add_order" in action.attrib:
            Verse += OrdersDict[action.attrib["add_order"]]
        elif "remove_order" in action.attrib:
            Verse -= OrdersDict[action.attrib["remove_order"]]
        elif "add_source" in action.attrib:
            Verse += SourcesDict[action.attrib["add_source"]]
        elif "remove_source" in action.attrib:
            Verse -= SourcesDict[action.attrib["remove_source"]]
        elif "save_state" in action.attrib:
            xmls.save_state(Verse, results_file=action.attrib["save_state"])
        elif "load_state" in action.attrib:
            BasePackDict, CombinedPackagesDict, NodesPackageDict, SimTimeHistory, HistoryIdx, NodePackageHistory = xmls.load_state(results_file=action.attrib["load_state"])
            Verse.SimTimeHistory = SimTimeHistory; Verse.SimTime = SimTimeHistory[-1]
            Verse.NodePackageHistory = NodePackageHistory
            nodelist = [NodesDict[label] for label in HistoryIdx.keys()]
            Verse.HistoryIdx = dict(zip(nodelist, HistoryIdx.values()))

            for label, val in BasePackDict.items():
                pack = PackagesDict[label]
                [HalfLifeDict, BranchingDict, MotherDaughterDict, DecayTypeDict, EnergyDict, NuclideDict,
                InvNuclideDict, InventoryStates, ActivityStates, HeatStates, HeatSumStates] = val
                pack.HalfLifeDict = HalfLifeDict; pack.EnergyDict = EnergyDict; pack.NuclideDict = NuclideDict; pack.InvNuclideDict = InvNuclideDict
                pack.BranchingDict = BranchingDict; pack.MotherDaughterDict = MotherDaughterDict; pack.DecayTypeDict = DecayTypeDict
                pack.InventoryStates = InventoryStates; pack.ActivityStates = ActivityStates; pack.HeatStates = HeatStates; pack.HeatSumStates = HeatSumStates

            for label, val in NodesPackageDict.items():
                node = NodesDict[label]
                recompiled_package_list = []
                for item in val:
                    time = item[0]
                    package_label = item[1]
                    if package_label in PackagesDict:
                        recompiled_package_list.append([time, PackagesDict[package_label]])
                    elif package_label in CombinedPackagesDict:
                        # Create a new CombinedPackage object and add it to the package list.
                        ## Convert the labeled PackageStates into objected PackageStates
                        for package_state, fac in CombinedPackagesDict[package_label]:
                            package_state[1] = PackagesDict[package_state[1]]

                        NewCombinedPackage = cge.CombinedPackage(PackageStates=CombinedPackagesDict[package_label], NuclideDict=NuclideDict, HalfLifeDict=HalfLifeDict, DecayTypeDict=DecayTypeDict)
                        NewCombinedPackage.label = label
                        recompiled_package_list.append([time, NewCombinedPackage])

                node.PackageList = recompiled_package_list
                node.TempPackageList = recompiled_package_list
                
        elif "organize_node" in action.attrib:
            ResultsDict = action.attrib
            
            resultvariables = action.find("variables")
            if resultvariables is not None:
                ResultsDict["variables"] = xmlc.split_text(resultvariables.text)
            elif "variables" in ResultsDict:
                pass
            else:
                ResultsDict["variables"] = None
                
            resultweights = action.find("weights")
            if resultweights is not None:
                ResultsDict["weights"] = xmlc.split_text(resultweights.text, cast=float)
            elif "weights" in ResultsDict:
                pass
            else:
                ResultsDict["weights"] = None
                
            resultopt = action.find("optimization")
            if resultopt is not None:
                ResultsDict["optimization"] = xmlc.split_text(resultopt.text, cast=float)
            elif "optimization" in ResultsDict:
                pass
            else:
                ResultsDict["optimization"] = None
                
            resultindif = action.find("indifference")
            if resultindif is not None:
                ResultsDict["indifference"] = xmlc.split_text(resultindif.text, cast=float)
            elif "indifference" in ResultsDict:
                pass
            else:
                ResultsDict["indifference"] = None
                
            resultpref = action.find("preference")
            if resultpref is not None:
                ResultsDict["preference"] = xmlc.split_text(resultpref.text, cast=float)
            elif "preference" in ResultsDict:
                pass
            else:
                ResultsDict["preference"] = None
                
            resultopt = action.find("optimization")
            if resultopt is not None:
                ResultsDict["optimization"] = xmlc.split_text(resultopt.text, cast=float)
            elif "optimization" in ResultsDict:
                pass
            else:
                ResultsDict["optimization"] = None
                
            resultfunctions = action.find("functions")
            if resultfunctions is not None:
                ResultsDict["functions"] = xmlc.split_text(resultfunctions.text)
            elif "functions" in ResultsDict:
                pass
            else:
                ResultsDict["functions"] = None
                
            if "size" in ResultsDict:
                elements = ResultsDict["size"].replace(" ", "").replace("(", "").replace(")", "").split(",")
                ResultsDict["size"] = np.zeros(tuple(int(x) for x in elements))
            else:
                print("*size* parameter needs to be inputted!")
                
            resultconf = action.find("configuration")
            if resultconf is not None:
                modified_list = (resultconf.text or "").replace("\t", "").replace(" ", "").split("\n")
                modified_list = [item for item in modified_list if item.strip()]
                
                # Convert mask into a NumPy array for easier processing
                mask_array = np.array([[char != '0' for char in row] for row in modified_list])
                mask_3d = np.broadcast_to(mask_array[:, :, np.newaxis], ResultsDict["size"].shape)
                # Apply mask to all depths (columns)
                ResultsDict["size"][mask_3d] = np.nan  # Broadcast over all depths
                
            SavedFileName = ResultsDict["organize_node"]
            ResultsDict.pop("organize_node")
            ResultsDict["filename"] = SavedFileName
            
            ResultsDict["node"] = NodesDict[ResultsDict["node"]]
            
            Verse.organize_node(**ResultsDict)
            
        elif "organize_packages" in action.attrib:
            ResultsDict = action.attrib
            
            resultvariables = action.find("variables")
            if resultvariables is not None:
                ResultsDict["variables"] = xmlc.split_text(resultvariables.text)
            elif "variables" in ResultsDict:
                pass
            else:
                ResultsDict["variables"] = None
                
            resultweights = action.find("weights")
            if resultweights is not None:
                ResultsDict["weights"] = xmlc.split_text(resultweights.text, cast=float)
            elif "weights" in ResultsDict:
                pass
            else:
                ResultsDict["weights"] = None
                
            resultopt = action.find("optimization")
            if resultopt is not None:
                ResultsDict["optimization"] = xmlc.split_text(resultopt.text, cast=float)
            elif "optimization" in ResultsDict:
                pass
            else:
                ResultsDict["optimization"] = None
                
            resultindif = action.find("indifference")
            if resultindif is not None:
                ResultsDict["indifference"] = xmlc.split_text(resultindif.text, cast=float)
            elif "indifference" in ResultsDict:
                pass
            else:
                ResultsDict["indifference"] = None
                
            resultpref = action.find("preference")
            if resultpref is not None:
                ResultsDict["preference"] = xmlc.split_text(resultpref.text, cast=float)
            elif "preference" in ResultsDict:
                pass
            else:
                ResultsDict["preference"] = None
                
            resultopt = action.find("optimization")
            if resultopt is not None:
                ResultsDict["optimization"] = xmlc.split_text(resultopt.text, cast=float)
            elif "optimization" in ResultsDict:
                pass
            else:
                ResultsDict["optimization"] = None
                
            resultfunctions = action.find("functions")
            if resultfunctions is not None:
                ResultsDict["functions"] = xmlc.split_text(resultfunctions.text)
            elif "functions" in ResultsDict:
                pass
            else:
                ResultsDict["functions"] = None
                
            if "sort" in ResultsDict:
                ResultsDict["sort"] = str(ResultsDict["sort"]).strip().lower() == "true"
            
            if "size" in ResultsDict:
                ResultsDict["size"] = int(ResultsDict["size"])
            else:
                print("*size* parameter needs to be inputted!")
                
            # Deal with criteria
            ordercriteria = action.find("criteria")
            if ordercriteria is not None:
                ordercriterialist = []
                for ordercriteria in action.findall("criteria"):
                    TempCriteriaDict = xmlc.convert_attrib(ordercriteria.attrib)
                    
                    if "limits" in TempCriteriaDict:
                        with open(TempCriteriaDict["limits"], "r") as file:
                            limits_dict = json.load(file)
                            TempCriteriaDict["limits"] = limits_dict
                            
                    criterianuclide = ordercriteria.find("nuclide")
                    if criterianuclide is not None:
                        crit = []
                        constraint = []
                        for crit_nuc in ordercriteria:
                            crit.append(crit_nuc.attrib["name"])
                            constraint.append(float(crit_nuc.attrib["criteria"]))
                        TempCriteriaDict["nuclide"] = crit
                        TempCriteriaDict["criteria"] = constraint
                    
                    ordercriterialist.append(TempCriteriaDict)
                    
                ResultsDict["criteria"] = ordercriterialist
            
            # Deal with extra conditions.
            extraconditions = action.find("conditions")
            if extraconditions is not None:
                extra_criteria = []
                for condition in extraconditions:
                    extra_criteria.append([x.strip().lower() == "true" for x in xmlc.split_text(condition.text)])
                ResultsDict["extra_criteria"] = extra_criteria
                
            SavedFileName = ResultsDict["organize_packages"]
            ResultsDict.pop("organize_packages")
            ResultsDict["filename"] = SavedFileName
            
            ResultsDict["node"] = NodesDict[ResultsDict["node"]]
            
            Verse.organize_packages(**ResultsDict)
            

        elif "save_results" in action.attrib:
            ResultsDict = action.attrib
            
            if "unify" in ResultsDict:
                ResultsDict["unify"] = bool(ResultsDict["unify"])
            
            resultnodes = action.find("nodes")
            if resultnodes is not None:
                ResultsDict["nodes"] = xmlc.split_text(resultnodes.text)
            elif "nodes" in ResultsDict:
                if ResultsDict["nodes"] == "all":
                    ResultsDict["nodes"] = [x.label for x in Verse.Nodes]
            else:
                ResultsDict["nodes"] = None
                
            resultpackages = action.find("packages")
            if resultpackages is not None:
                ResultsDict["packages"] = xmlc.split_text(resultpackages.text)
                if bool(GroupPackagesDict):
                    for lidx, lbl in enumerate(ResultsDict["packages"]):
                        if lbl in GroupPackagesDict:
                            ResultsDict["packages"][lidx:lidx+1] = GroupPackagesDict[lbl]
                        
            elif "packages" in ResultsDict:
                if ResultsDict["packages"] == "all":
                    ResultsDict["packages"] = [x.label for x in list(Verse.UniquePackageDict.values())]
            else:
                ResultsDict["packages"] = None
                
            resultvariables = action.find("variables")
            if resultvariables is not None:
                ResultsDict["variables"] = xmlc.split_text(resultvariables.text)
            elif "variables" in ResultsDict:
                pass
            else:
                ResultsDict["variables"] = None
            
            resultnuclides = action.find("nuclides")
            if resultnuclides is not None:
                ResultsDict["nuclides"] = xmlc.split_text(resultnuclides.text)
            elif "nuclides" in ResultsDict:
                pass
            else:
                ResultsDict["nuclides"] = None
            
            resulttimes = action.find("times")
            if resulttimes is not None:
                ResultsDict["times"] = xmlc.split_text(resulttimes.text, cast=float)
            elif "times" in ResultsDict:
                pass
            else:
                ResultsDict["times"] = None

            SavedFileName = ResultsDict["save_results"]
            ResultsDict.pop("save_results")

            results_data = Verse.get_results(**ResultsDict)

            if ResultsDict["nodes"] is None:
                old_column_list = results_data[2][1]
                if resultpackages is not None:
                    column_list = ResultsDict["packages"]
                else:
                    column_list = list(PackagesDict.keys())
            else:
                old_column_list = results_data[2][0]
                column_list = ResultsDict["nodes"]
                if isinstance(column_list, str):
                    column_list = [column_list]

            # Each variable is either "simple" (one number per node/package per timestep --
            # mass, volume) or "per_nuclide" (one number per node/package PER NUCLIDE per
            # timestep -- activity, heat, etc). The old Excel format flattened the per_nuclide
            # case into stacked column blocks within a sheet, with no marker for where one
            # block ends and the next begins other than a blank header cell -- fragile to
            # parse back out reliably. Representing it as actual nested keys instead (series
            # label -> nuclide label -> time series) says exactly what the data is with
            # nothing to reconstruct.
            output = {"time": [float(t) for t in results_data[0]], "variables": {}}

            for i, slice_ in enumerate(results_data[1]):
                variable_name = ResultsDict["variables"][i]
                order = [old_column_list.index(lbl) for lbl in column_list]
                slice_ = slice_[order]
                slice_ = slice_.T

                if len(slice_.shape) == 1:
                    slice_ = np.expand_dims(slice_, axis=1)

                if len(slice_.shape) == 3:
                    # shape is (nuclide, time, series)
                    series_dict = {label: {} for label in column_list}
                    for j in range(slice_.shape[0]):
                        nuclide_label = results_data[3][j]
                        for k, label in enumerate(column_list):
                            series_dict[label][nuclide_label] = slice_[j][:, k].tolist()
                    output["variables"][variable_name] = {"kind": "per_nuclide", "series": series_dict}
                elif len(slice_.shape) == 2:
                    # shape is (time, series)
                    series_dict = {label: slice_[:, k].tolist() for k, label in enumerate(column_list)}
                    output["variables"][variable_name] = {"kind": "simple", "series": series_dict}

            with open(SavedFileName, "w") as f:
                json.dump(output, f)