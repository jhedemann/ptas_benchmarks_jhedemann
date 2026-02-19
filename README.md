# Online Slow Wave Detection and Phase-Targeted Stimulation

## Description

This repository harbors an adaption of the TWave algorithm (Li et al., 2025), optimized for online use on intracranial EEG.

## Features

*   **Phase-Tracking Algorithm**: Includes implementation TWave, which detects slow waves, schedules stimulations at up-state. 
*   **Plotting**: Includes plotting functionalities from the original [repository](https://github.com/gmilab/ptas_benchmarks/blob/main/Algo_TWave.py).
*   **Quality Metrics**: Includes various scripts to quantify how well the algorithm a) detects slow waves, b) rejects IEDs, and c) stimulates at the targeted phase.
*   **Jupyter Notebook for Workflow**: An example notebook (`run_group_simulations.ipynb`) demonstrates a typical workflow for running group simulations and generating results.

## Repository Structure

*   `Algo_TWave.py`: Python file implementing TWave algorithm.
*   `Simulations.py`: Core Python module containing simulation logic, data loading functions (e.g., `load_anphy_data`, `get_anphy_datasets`), and plotting utilities.
*   `run_twave.py`: Main Python file for running the algorithm.
*   `Inhibitors.py`: Module defining inhibitor classes that can be used by the algorithms to control stimulation.
*   `plot/`: Folder containing different scripts for plotting and quality metrics.
*   `utils/`: Folder containing different utilities like `analyze_time_frequency.py` and `load_intracranial_data.py`.

## Requirements

This repository assumes that EEG files as well as ground-truth annotations for slow waves and IEDs are present within the same folder, e.g. data/annotated/. Many scripts depend on this, apart from the core algorithm and run script.

## Example Workflow

1. Run the TWave algorithm using `run_twave.py`.
2. Visually inspect the results using `load_and_test_results.py`.
3. Compute quality metrics using `compute_detection_quality.py`.
4. Plot an overview using `plot_master.py`.