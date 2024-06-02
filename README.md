# Continuing to work on Modelo
Modelo is the model I made when I participated in the AI data readiness challenge for the National Cancer Institute in April of 2024.
It is a binary classifier that was trained on data from the National Lung Screening trial. 
It can predict whether Cancer is present or not in a CT lung cancer scan. 

After the competition was over I revisited the model to see what I could improve.
This is the result of that work. 


The original model:
<b>Metric results</b> 
 * Accuracy 0.5305555661519369 
 * Recall 0.5879966815312704 
 * F-1 Score 0.6082125902175903 
 * Precision 0.6471310456593832

 After my changes:
 <b>Metric results</b> 
 * Accuracy 0.733010
 * Recall 0.85185
 * F-1 Score 0.7329
 * Precision 0.77500
 

The original model can be found at the National Cancer Institute:
https://computational.cancer.gov/model/aidr-challenge-tier-1-mcfarlin
and this repo:
https://github.com/oohtmeel1/AI-Data-Readiness-Challenge-for-the-NCI-Cancer-Research-Data-Commons

Requirements and usage:
In order to be able to use the model you will need the following:

The model takes about 512mb of space per forward pass. 


`pytorch 2.1.0` (requirements.txt should take care of that)

`python 3.11.7` 

`folder of JPEG images` <- A single folder of JPEG images. Containing both positive and negative image files of lung cancer. I used DICOM2jpeg to convert the files. 

`train` `val` `test` <- CSV files containing training labels. 

`loading_data_files.py` <- Dataloader file, uses Pytorch.

`defining_directories.py` <- defines file locations otherwise it will run using pre-specified defaults.

`model_architecture.py` <- File containing the various layers of my model.

`begin_experiment.py` <- File to run the experiment. Creates a tensorboard directory to log data, and saves models at several checkpoints. 
saves resulting metrics to a csv `result_file.csv`. 

Metric Calls 
After the model makes a prediction it will compute the following metrics:

`recall`,`accuracy`,`F1`,`precision`

Type the following into your terminal to call them:

`acc`  <- Binary Accuracy

`recall` <- Recall

`F1` <- F1 Score

`precision`<- Binary Precision

Additionally a csv file will be saved that will contain the names of the best models, and the metrics from above. 





