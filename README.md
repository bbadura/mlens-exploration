# ople-project
Version control for ensembling project

To run the program, simply clone this repository onto your local machine and run "make". The makefile will spin up a docker and the program will be able to run without having to install dependencies on your own machine.

Before running the program, it is important to format the dataset to be understood by this program. The single version of this application will run a single dataset and output a table, which can be compared against an assortment of other datasets. When uploading the dataset, you will need to specify the target field, as well as note if there are designated header names or not. The default target class in this is 559, as that is the target class of the included dataset pair, obtrain.csv and obtest.csv. Simply change this number to a different number or string to reflect your target class. Additionally, if headers are included, you will need to remove the "headers=None" parameter from the pandas load call.

Once you have formatted the data upload, you can choose to add as many function calls as you would like to try out different ensemble methods. Add these below "Function calls to create and test ensembles" comment. Be sure to observe the inputs for each function call, and use the defaults as an example for formatting.

The output is formatted in a 2-D chart and shows accuracy and the amount of time it took to run each specific ensemble (and test it). For cross validation results (to test for consistency), simply set the flag for this in the function call. Otherwise, the program will output "N/A" in the cross validation field.

Be patient when program is running. Depending on the number of models you run and the size of the datasets, the program can take upwards of 10 minutes to complete. In the future, there will be more detailed status updates to allow the user to know where the progress of the program is.
