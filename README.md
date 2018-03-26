# ople-project
Version control for ensembling project

To run the program, simply clone this repository onto your local machine and run "make".
The makefile will spin up a docker and the program will be able to run without having to install dependencies on your own machine.
The current version only supports the obtrain/obtest datasets but in the future will allow for customization.

Before running the program, it is important to format the dataset to be understood by this program.
The single version of this application will run a single dataset and output a table, which can be compared against an assortment of other datasets.
When uploading the dataset, you will need to specify the target field, as well as note if there are designated header names or not.

Be patient when program is running. Depending on the number of models you run and the size of the datasets, the program can take upwards of 10 minutes to complete.
