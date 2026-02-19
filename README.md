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
*   `run_group_simulations.ipynb`: An example Jupyter Notebook demonstrating how to run simulations across multiple subjects and algorithms, and how to generate group-level results and plots.

