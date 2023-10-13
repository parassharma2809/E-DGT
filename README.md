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

## Running the Models
* Change the model to your selected model in the train.py file
* Update any specific configuration in the corresponding functions in the file.
* Run ```python train.py```
* This is by default set in the debug mode. To remove the debug mode and run the full training, remove the ```--debug``` from the sys config in the file.
* There might be path errors for the data. Just make sure before running the file that all the paths are correct with respect to the train file.
* Also, training is set in CPU. Just change the mode to GPU in the train function and it'll run on the default GPU on your device.
* If you want to use multiple GPUs, setup the code accordingly.