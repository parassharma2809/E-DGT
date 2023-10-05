# E-DGT

## Setup the Repository
* Clone the Repo.
* Inside the E-DGT root folder make a data directory and copy all the data folders in this directory.
  * Audio, frames, qa, transcript, video
* Make a virtual python 3.9 environment
* Run pip install -r requirements.txt

## Running Preprocessing Commands
* Navigate to preprocess directory
  * ```cd preprocess```
* Run ```python config.py```
* If it runs successfully, then run ```python dataset.py```
* These should run successfully if the environment has been set up correctly and the data is on correct path.