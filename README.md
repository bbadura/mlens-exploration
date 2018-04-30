# MLEns Exploration
Version control for ensembling project

This GitHub provides neccesary building componets, test data sets, and results for the intern project regarding MLEnsemble, a python package for ensembling.

To run the program, simply clone this repository onto your local machine or onto an AWS machine. If you want to run this is Docker, run "make test" and you can run the program without having to download dependencies. If you want to run locally or on AWS, pip install the following packages: pandas, numpy, sklearn, mlens, texttable. Then, run "make local".

Before running the program, it is important to format the dataset to be understood by this program. This application will run a single dataset and output a table, which can be compared against an assortment of other datasets (I chose to use Excel for this portion). Follow the instructions below to make sure that when you run the program, it is doing what you want it to do.

1. Figure out which program to run! For datasets with both a train and a test set, use ensemble_1.3. For datasets that do not have a test set, use ensemble_notest. This method will use cross validation instead of comparing on a test set.
2. Look for all lines that have "###### EDIT ######" after them... these are lines that need be editted to customize your request. Steps 3- will describe what each of these requires, going from top to bottom
3. iters: number of iterations to run (average of multiple iterations is the output of the program)
4. files: data file/files to upload
5. train/test_df: creating the dataframe. For datasets with no headers, include "header=None" in the read_csv call
6. file_output: the output file name. Name accordingly but make sure you don't overwrite other files
7. dataset['--']: Maps variables for classification. The index should be the target class index (a word if there is a header, a number if the header is not specified). Map the binary classes to integers
8. X/Y_train/test: simply change the first argument to the name of the index for each of these.
9. Feature Selection: If your dataset doesn't have header names and has more than 20 classifiers, uncomment this block of code
10. for j in range(0,15): number of models to compute
11. ('dtc', DecisionTreeClassifier(....: choose and name your ensemble input elements.
12. output[i]['super_dtc'] = add_superlearner(: name the elements properly for the dictionary

You can choose to add as many function calls as you would like to try out different ensemble methods. Add these in the same area that the other function calls are in (nested in the for loop) and name them accordingly. Be sure to observe the inputs for each function call, and use the defaults as an example for formatting.

The output is formatted in a 2-D chart and shows accuracy and the amount of time it took to run each specific ensemble (and test it). Additionally, it saves the output that you see on the screen to the filename that you specify in the program. You can choose not to save by simply setting this variable to "None".

***BEFORE RUNNING: Change the Makefile and Dockerfile to properly reflect the functions and datasets you want to reference. If you fail to do this, the program will not run properly.***

Be patient when program is running. Depending on the number of models you run and the size of the datasets, the program can take upwards of 10 minutes to complete. The program will print out brief status updates, but they are more for checkpointing purposes and not accurate description of percentage complete.

# FAQ
**How do I run a test one data that doesn't have a specified test set?**
*To run this program on a data without a test set, use the ensemble_notest.py file... this will use cross-validation as the primary source of calculating accuracy scores*

**Why am I coming up with results that give and accuracy score of 1?**
*A possible reason for this may be because you are using a dataset that has very few attributes. Dataset with few attributes typically don't perform very well*

**Can I run more than one dataset at once?**
*This program is only set up to run one dataset at a time (a train set and a test set are considered a single unit). To run multiple tests, you will have to run multiple instances of the program.*

**I'm getting an error that says the number of features is too high...**
*You need to change the random range in the num_features attribute to be at most the number of features that you are feeding into the models* 
