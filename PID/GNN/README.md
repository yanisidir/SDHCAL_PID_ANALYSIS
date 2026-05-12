# GNN PID

This folder contains exploratory graph-neural-network studies for SDHCAL PID using PyTorch Geometric.

## Main Scripts

- `GNN.py`: graph-based PID workflow for a two-class configuration.
- `GNN_3_classes.py`: graph-based PID workflow for pi-, K0, and proton.
- `GNN_QI.py`, `GNN_W.py`, `GNN_debug.py`, `GNN_PID_old.py`: variants and development scripts.

## Method

The scripts build graph-like event representations from hit or flattened CSV information, apply spatial and timing preprocessing, and train graph neural networks using layers such as GraphConv, TAGConv, or GATConv.

## Outputs

- `plots/`: loss/accuracy and confusion-matrix plots from previous runs.
- `GNN_performances.csv`, `run_parameters.csv`, `run_comments.csv`: run logs.
- generated `.pt` model checkpoints are ignored for future commits.

These studies are exploratory and should be interpreted as a research direction rather than the final PID baseline.
