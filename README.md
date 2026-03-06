# transchem+
# Approach

![描述文字](fig_main2.png)

Overview of the proposed GraphMD. We encode temporal trajectories using the Discrete Cosine Transform (DCT) to extract dominant low-frequency motions. And we further incorporate guiding terms based on Morse and Lennard-Jones (LJ) potentials to impose soft physical constraints during graph processing, enabling the model to integrate meaningful physical priors and obtain effective guidance before adding noise during sampling.

# Datasets
2. The datasets used in the experiments are located in the following folder path:

```
root/
├── transchemplus/
│ └── data/
```


# Dependencies
we recommend installing the following packages:

```
python == 3.12
torch == 2.10.0
torchvision == 0.25.0
torch-geometric == 2.7.0
torch_scatter == 2.1.2
numpy == 2.4.2
tqdm == 4.67.3
pandas == 3.0.1
sklearn == 1.8.0
rdkit == 2025.9.6
```
