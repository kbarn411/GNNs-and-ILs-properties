# GNNs and ILs properties - cheminformatics study

This repo contains codes and data to reproduce our work on Graph Neural Networks and structural information on ionic liquids: cheminformatics study on molecular physicochemical property prediction. 

Corresponding author: Karol Baran (GdańskTech), karol.baran[at]pg.edu.pl

Manuscript status: accepted Journal of Physical Chemistry B - Machine Learning in Physical Chemistry Virtual Special Issue (2023) 

## Project description

Graph neural networks (GNNs) have emerged as a powerful cheminformatic tool, but their application to more complex classes of compounds, such as ionic liquids, has been limited. In this work, we critically evaluate the use of GNNs in structure-property studies for predicting the density, viscosity, and surface tension of ionic liquids. We discuss the problem of data availability and integrity, and demonstrate how GNNs can be used to deal with mislabeled chemical data. We discuss if providing more training data could be more important than ensuring that the data is immaculate. We also pay close attention to how GNNs process graph transformations and electrostatic information. Our results provide guidance on how GNNs should be applied to predict the properties of ionic liquids.

## HOW-TO 

- clone this repo

`git clone https://github.com/kbarn411/GNN-physchem-ILs.git <your_directory>`

- prepare conda environment

`$ conda create --name <env> --file requirements.txt`

- change directory to `data`

`cd ./data`

- download datasets

        Datasets are from works by Paduszyński:
        1. density - Paduszyński, K. (2019). Industrial & Engineering Chemistry Research, 58(13), 5322-5338.
        2. viscosity - Paduszynski, K. (2019). Industrial & Engineering Chemistry Research, 58(36), 17049-17066.
        3. surface tension -  Paduszynski, K. (2021). Industrial & Engineering Chemistry Research, 60(15), 5705-5720.
        
        In order to proceed please download and extract files from the following links:
        1. density - https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.9b00130/suppl_file/ie9b00130_si_001.zip
        2. viscosity - https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.9b03150/suppl_file/ie9b03150_si_001.zip
        3. surface tension - https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.1c00783/suppl_file/ie1c00783_si_001.zip
        
        After compliting the stage please rename them to:
        1. density.xlsx
        2. viscosity.xlsx
        3. surface_tension.xlsx

        and run
        
        python3 preparedata.py

- change directory to `code`

`cd ../code`

- run file `experiment.py`

`python3 experiment.py`

