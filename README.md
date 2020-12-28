# Course Project


Each folder contains code related to separate parts of the project, RSO to the part on Random
Search Optimization, and TS to the part on Thompson Sampling. 

## RSO

### main.py
This code runs the various versions of RSO, including the original implementations as well as my own.
It also loads in and potentially normalizes different datasets, including the three in the write-up.
In order to use different update sampling methods, simply change the delta_type argument (0, 1, or 2).
Most other modifications are desribed in at the bottom of the code, below the data loader functions.
I didn't include a modular method of swapping the loss function as I did that part manually, but if
you'd like to see that part it's as simple as changing the criterion in the training function and then
adding an argmax to the getLoss function (though I figure you're familiar with this type of thing anyway). 

### RSO_updates.py
This code is the meat of this section, containing the baseline model, the three versions of RSO updates,
and the training scheme. See the comments and the write-up itself for an explanation of each component.

### base_rso_plotting.py and binary_rso_plotting.py
These are just files to succinctly plot the results of my experimentation with RSO and variants
as shown in the report itself. It should be noted that the code expects that data files to be named
certain things in particular, so if you'd like to run this code you'll need to first generate the 
data using those names and place them in a folder with the necessary name (or, change the directory
reference).

### standard_updates.py
Contains the functions and classes needed to load and train a standard network via SGD, for comparison.
I've not updated this code in a while except to add new dataset baseline model overrides as I did not
compare to SGD too much, but I left the code in anyway in case you wanted to see this.

### BaseNetworks.py
This file constains all of the baseline models for the different datasets. It allows for easy swapping
of model structure if needed.



## TS

### main.py
This file is very similar to the main code in RSO, however you can also control forgetting, annealing 
schedule, and Thompson Sampling type in the arguments of the training function call. See descriptions
of each in the docstring at the end of the code in this file or in the training function in the next 
file.

### TS_Updates.py
This code contains the baseline model, Thompson Sampling update versions, training function, and related 
functions such as those that produce performance metrics and the belief structure. See the code comments
as well as the write-up for a description of each component.

### "plotting.py" files
Similar to the RSO plotting files, just for Thompson Sampling results.

### BaseNetworks.py
Should be almost exactly the same as the identically named code in the RSO folder, but sometimes these
parts of the project were in separate locations so I keep two versions.
