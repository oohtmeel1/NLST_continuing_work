# Continuing to work on Modelo
Modelo is the model I made when I participated in the AI data readiness challenge for the National Cancer Institute in April of 2024.
It is a binary classifier that was trained on data from the National Lung Screening trial. 
It can predict whether Cancer is present or not in a CT lung cancer scan. 

After the competition was over I revisited the model to see what I could improve.
This is the result of that work. 


The original model:
<b>Metric results</b> 
 * Accuracy 0.53055
 * Recall 0.58799 
 * F-1 Score 0.60821 
 * Precision 0.64713

 After my changes:
 <b>Metric results</b> 
 * Accuracy 0.73301
 * Recall 0.85185
 * F-1 Score 0.73290
 * Precision 0.77500
 

The original model can be found at the National Cancer Institute:
https://computational.cancer.gov/model/aidr-challenge-tier-1-mcfarlin
and this repo:
https://github.com/oohtmeel1/AI-Data-Readiness-Challenge-for-the-NCI-Cancer-Research-Data-Commons

Requirements and usage:
In order to be able to use the model you will need the following:
Files will be in the folders of this repo if available. 

The model takes about 512mb of space per forward pass. 

`pytorch 2.1.0` (requirements.txt should take care of that)

`python 3.11.7` 

`folder of JPEG images` <- A single folder of JPEG images. Containing both positive and negative image files of lung cancer. I used DICOM2jpeg to convert the files. 
If you want to demo the model you can download all of the transformed jpeg images at this link (They require about 1GB of space). https://drive.google.com/drive/folders/1MLwxhcQmn7qXqy_iP2zDjLGyxa_G_jLS


`train` `val` `test` <- CSV files containing training labels. Demo files are located in the files folder of this repo. 

`loading_data_files.py` <- Dataloader file, uses Pytorch.

`defining_directories.py` <- Defines file locations.

`model_architecture.py` <- File containing the various layers of my model.

`begin_experiment.py` <- File to run the experiment. Creates a tensorboard directory to log data, and saves models at several checkpoints. 
saves resulting metrics to a csv `results.csv`. 


Metrics

After the model makes a prediction it will compute the following metrics:

`recall`,`accuracy`,`F1`,`precision` and the predicted and true labels. In case other calculations are needed. 


Additionally a csv file will be saved that will contain the names of the best models, and the metrics from above. I'll add more as I work on it.



Also here is the Citation for the images from the NCI
National Lung Screening Trial Research Team. (2013). Data from the National Lung Screening Trial (NLST) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.HMQ8-J677




