https://gram-competition.github.io/
https://github.com/gram-competition/iclr-2026
https://huggingface.co/datasets/gram-competition/warped-ifw

This year’s competition hosted in conjunction with the Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM) will be a benchmark challenge. We have prepared a dataset of 3D geometries inspired by the front wing of a Formula 1 car for which BeyondMath kindly provided transient simulations of airflow specifically set-up for an academic-style challenge.

https://gram-competition.github.io/assets/front_wing.png
https://gram-competition.github.io/assets/airflow.mp4

The challenge is about generating airflow at future time points based on the geometry and airflow at previous time points. The winner of the competition is going to receive the MCML Award consisting of 500 € in prize money. Furthermore, we are going to publish a description of the challenge and all valid submissions in the workshop proceedings and participants will have the option to be co-authors.

Submissions will take the form of pull requests to our GitHub repository (link above) and you can participate as a team. For questions, open an issue on GitHub or send us an email.

Deadline is on April 22, 2026 (AoE).

Challenge
Consider a 3D velocity field 𝑢⁡(𝑡,𝑥) of airflow around an airfoil where 𝑡0 ≤𝑡 ≤𝑡1 and 𝑥 ∈Ω ⊂ℝ3. Denote the airfoil surface by 𝜕Ω and assume no-slip boundary condition:
𝑢⁡(𝑡,𝑥)=(0,0,0)𝖳for𝑥∈𝜕Ω.
Given 𝑢⁡(𝑡,𝑥) for 𝑡0 ≤𝑡 ≤𝑡𝟎.𝟓 our goal is to estimate 𝑢⁡(𝑡,𝑥) for 𝑡𝟎.𝟓 <𝑡 ≤𝑡1. In other words, we are looking for a neural operator 𝐺𝜕Ω⁡(𝑡,𝑥) conditioned on the geometry of the airfoil 𝜕Ω that predicts the velocity field at the following time points based on the preceding ones.

In practice, this can be any model that takes as input a discrete velocity field at a fixed number of time points and points in space, as well as the airfoil geometry. Graph neural networks or transformers could be suitable models for this task.

Aerodynamics usually decompose into low-frequency (“laminar”) and high-frequency (“turbulent”) components. The preceding velocity field already provides an excellent prior for the low-frequency components of the following dynamics. For this reason, we expect the main difficulty of this challenge to be estimation of high-frequency components in the airflow.

Dataset
Our dataset consists of 3D geometries made up of one, two or three differently-sized airfoils at randomly-sampled relative positions and pitch angles, thereby spanning a rich space of geometric variation.† The airfoil geometry is derived from the Imperial Front Wing (IFW), a Formula 1-style front wing CAD geometry developed at Imperial College London [Buscariolo, 2019].



Airflow was simulated based on a constant freestream velocity in 181 of these geometries and we extracted five time windows from each simulation for our dataset. In order to make the simulation results easier to work with, we subsampled a fixed number of points from the velocity field. The airfoil surface is encoded as a subset of those points.

Samples (annotated by their dimensions) are saved as follows.

Dataset
├── "1021_1-0"                 "<simulation ID>-<index of time window>"
│   ├── t: (10,)
│   ├── pos: (100k, 3)
│   ├── idcs_airfoil: (20k,)      indexing pos with values in [0, 100k)
│   ├── pressure: (10, 100k)          provided for sake of completeness
│   ├── velocity_in: (5, 100k, 3)
│   └── velocity_out: (5, 100k, 3) 
│
├── ...
│
└── "3006_17-4"
    ├── t: (10,)
    ├── pos: (100k, 3)
    ├── idcs_airfoil: (8282,)             number differs per simulation
    ├── pressure: (10, 100k)
    ├── velocity_in: (5, 100k, 3)
    └── velocity_out: (5, 100k, 3)
The dataset can be downloaded from Hugging Face (link above).

Ranking
Submissions will be evaluated on a held-out test split. This means that all available data can be used by participants for training. We do not disclose the specific evaluation metric, but it will measure accuracy, i.e., similarity of the estimated 3D velocity fields and the ground truth.

Disclaimer
The aim of the challenge was to have a task involving sequential point-cloud prediction. Fluids were the first thing that came to mind. We’ve sliced a small part of the IFW and then scaled it anisotropically to create a new set of warped geometries. The data is designed in this specific way for the workshop competition (GRaM) rather than a fully experimentally validated CFD reference dataset. We targeted a consistent 𝑦+ regime and a lightweight setup, balancing accessibility for workshop participants with suitability for popular learning-based methods and comparative evaluation, i.e., keeping the cell count low but academically interesting enough in the broader context of what the workshop is about. This allows focus on the development of the geometric method.

The aim is to explore learning for the task of sequential point-cloud data on a fun and interesting geometrical set of variations.

Organisers
Alison Pouplin
Alison Pouplin
Bayer
Gavin Seegoolam
Gavin Seegoolam
BeyondMath
Julian Suk
Julian Suk
TU Munich