# Event Camera Segmentation Experiments

This repository contains the code I used for my event-camera segmentation experiments. The goal is to test how much scene information can still be learned from event-camera data using different event representations.

The project has two main experiments:

* **DSEC experiment**: semantic segmentation on DSEC-style driving data.
* **PEDRo experiment**: binary human segmentation using PEDRo bounding-box labels as pseudo masks.

## Project structure

```text
segmentation_experiment/
├── dsec_experiment.py          # DSEC segmentation experiment
├── pedro_experiment.py         # PEDRo human-box segmentation experiment
├── event_representations.py    # event representations used by both experiments
├── segmentation_model.py       # segmentation model
├── requirements.txt            # Python dependencies
├── dsec_seg/                   # put the DSEC dataset here
├── pedro/                      # put the PEDRo dataset here
└── runs/                       # created automatically when experiments are run
```

The `dsec_seg/` and `pedro/` folders are intentionally included as empty dataset folders. The datasets themselves are not included in this repository.

## Setup

This project was tested with Python 3.10 or newer.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Datasets


### DSEC

Put the DSEC data inside:

```text
dsec_seg/
```

The expected structure is:

```text
dsec_seg/
├── zurich_00/
├── zurich_06/
├── zurich_07/
├── zurich_08/
└── zurich_13/
```

### PEDRo

Put the PEDRo data inside:

```text
pedro/
```

The expected structure is:

```text
pedro/
├── numpy/
│   ├── train/
│   ├── val/
│   └── test/
└── xml/
    ├── train/
    ├── val/
    └── test/
```

The `.npy` files contain the event data, and the `.xml` files contain the bounding-box annotations.

## Event representations

The experiments use event representations from `event_representations.py`, mainly:

* `recent`
* `voxel`
* `evsegnet`

## Running the experiments

Run the DSEC experiment with:

```bash
python dsec_experiment.py
```

Run the PEDRo experiment with:

```bash
python pedro_experiment.py
```

The scripts use the settings defined near the top of each file. Change the dataset path, number of epochs, batch size, seeds, or representations there if needed.

## Outputs

Experiment outputs are written to:

```text
runs/
```

The most useful files are:

```text
summary_results_by_seed.csv
summary_results_mean_std.csv
```

For PEDRo, the script also saves tables such as:

```text
all_metrics_by_seed_and_window.csv
mean_std_by_window.csv
compact_table.md
```

Model checkpoints are saved as `.pt` files inside the run folders.

## Notes

* PEDRo labels are bounding boxes, not exact human silhouettes. Therefore, the PEDRo human IoU should be understood as **box-mask IoU**, not true silhouette IoU.
* DSEC labels are remapped into broader categories such as flat, background, object, vegetation, human, and vehicle.
* The code uses random seeds for repeatability, but exact results can still vary slightly depending on hardware and PyTorch/CUDA versions.

