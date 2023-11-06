# Detecting changes of sound pressure level for automatic volume control using EEG 

Code for Kimura et al. 2023

## Reference

An auditory brain-computer interface to detect changes in sound pressure level for automatic volume control

Kimura et al. 2023

## Data

Preprocessed data are available from the following form.

[Request form](https://forms.gle/wgHoyvMPR4gwCP5a7)

### exp1

- 7 subjects
- subject-1-XX.mat (XX is ID number)
- Data format (variables)
  - DataF: Data, 1300x256x64 (trial x time x channels)
  - Pressure_Label: sound pressure labels, 1300 x 1 (trial x 1)
    - 1: target (70 db)
    - 2: non-target (60 db)

### exp2

- 10 subjects
- subject-2-XX.mat (XX is ID number)
- Data format (variables)
  - DataF: Data, 1300x256x64 (trial x time x channels)
  - Pressure_Label: sound pressure labels, 1300 x 1 (trial x 1)
    - 1: target (70 db)
    - 2: non-target (60 dB)
    - 3: target (50 dB)
  - Group: 1300 x1
    - 1: target (70 db)
    - 2: non-target

## Analysis

- Prepare the data in /data/exp1 or exp2
- Run main.py

## Requirement

- Python 3.X 
- other packages/tools [(requirement.txt)](./analysis/requirements.txt)

## NOTE

The code is used for the analysis after preprocessing.
