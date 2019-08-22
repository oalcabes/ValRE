# ValRE

ValRE validates spaceweather forecasting models of SEP events. Validation is done by
comparing model forecasts of events to historical observations and generating skill scores,
which are written into reports (in PDF and/or JSON format). ValRE also creates plots
for qualitative verification, which appear in the PDF and are present as PNG files in the
validation_reports/figures folder.

***NOTE: ValRE was not intended for the creation of model output or observational output. You must already have output present in some directory on your computer in order to run ValRE.

ADDITIONALLY: IF YOU NEED TO ALTER THE CODE, PLEASE READ ValRE_ADDITIONS.TXT FIRST

Included With the ValRE Package:
================================
ValRE_README.txt  
ValRE_ADDITIONS.txt  
ValRE.py  
config.py  
operational_sep_quantities.py  
output_to_json.py  
gen_output.py  
validation_reports (inner directory - may not be present until first run)  
validation_reference.pdf (will be created with first ValRE run, reference sheet for metrics)
validation_reports/figures (inner directory that will house PNG files)  
GOES_primary_assignments.csv  
instrument_dates.csv  

Python Libraries
================
Before running ValRE, you must make sure you have the correct Python libraries loaded on your computer.
Most of these libraries are included in Anaconda Navigator. However, it may be necessary to install some
yourself. In order to do so, type: "pip install <library>".
   
Libraries you will likely have to install:
pip install wget
pip install reportlab

The full list of modules that ValRE uses can be found at the beginning of the ValRE.py code.

This program makes use of the software "operational_sep_quantities" created by Katie Whitman, a program
which has extensive that can be found with the command "pydoc operational_sep_quantities".

There are two other scripts included with the ValRE package, gen_output.py and output_to_json.py, which are scripts that can be used to 
generate observational output from operational_sep_quantities.py, and each have their own documentation comments within the code.

Assumptions/Simplifications
===========================
please be aware of all assumptions and simplifications ValRE makes before running the code so you do not get unclear or incorrect
output.
- all output files have some string of the date in them in the forms: YYYY-MM-DD or YYYY_MM_DD
- model file start and date times are the model's PREDICTION WINDOW. That is, the model is predicting an all clear for that given time 
  period
  **note: if model start and end times are the same, that is the same as saying there has not been a prediction
- all output files are in the JSON form created by the CCMC (this example format can be found in
  ref_files/example_sepscoreboard_json_file_v20190228.json - however, please do not move/alter this file from its current directory as
  ValRE or one of the other functions may need to read it in. If you need to use it please just copy it)
- there is one observation file per event
- all JSON files present in given output directories (and all subdirectories) are to be used for validation (if there are files you
  don't want to use for validation, put them in a different folder)
- when extracting forecast values for peak flux or probability, if a threshold is crossed, ValRE will use the first probability or flux
  value that crosses the threshold as opposed to the highest value for mean percent error and mean absolute percent error calculations,
  as well as for plotting purposes. if no threshold is crossed, ValRE will use the highest probability or flux value forecasted for a 
  given event.
- ValRE will only use flux and probability thresholds given in the configuration file. If your model output has a different threshold   
  than that given in the configuration file, ValRE will not read it or use it. 

Usage instructions
==================
Once you are sure you have correct model output, open config.py in your python editor of choice - Vi, Vim, or Spyder are just a few 
examples of options if you don't already have one.

In the configuration document, specify:
1. the directory on your computer containing model output
2. the name of the model
3. the directory on your computer containing observational output (or the directory where you would like
   new observational output to be stored)
4. the date for the first day you are interested in validating, separated by year, month, and day
5. the date for the last day you are interested in validating, separated by year, month, and day
   **NOTE: you do not actually have to have model output for your beginning and end dates. ValRE will
   simply extract all model files that are dated within the date range that you have given. Again, the
   date MUST be in the name of the model file!! *(note to self - talk to phil abt this prob)
6. detect previous event value ** explained in detail below, please read before changing from False
7. thresholds for the energy, the flux, and/or the probability
8. whether or not you'd like PDF and JSON reports
9. manual inputs for hits, misses, correct negatives, and false alarms

The formats in which all of these values must be given in are detailed in the comments of the
configuration file, as well as in the "input formats" section below.

Once you've filled out the configuration file, you can run ValRE from the command line. Once ValRE is finished
running, you can find your reports in the validation_reports directory, and PNGs of your figures in the figures
validation_reports/figures directory.

A reference sheet called validation_reference.pdf for how to understand all of the calculated metrics is included with the code, and 
will be created in the validation_reports directory the first time the code is run.

Input Formats
=============
PATHFILES
---
FORMAT: output directories in the form of a string  
EX. '.\Mag4_output'  
OTHER NOTES: Write the directory however you would on your own computer.
The current working directory is the directory you have placed the ValRE package in.
If you do not have dates specified in the names
of your model files, ValRE will NOT be able to function

MODEL NAME
---
FORMAT: string  
EX. 'MAG4'  
OTHER NOTES: this will only be used for report-writing purposes

INSTRUMENT FOR OBSERVATIONS
---
FORMAT: string  
EX: 'GOES-08'  
OPTIONS: GOES-08, -10, -11, -12, -13, -14 -15, SEPEM  
OTHER NOTES: ValRE automatically chooses the correct GOES observational instrument to
use for your given start and end dates. However, if you'd like to specify one instrument
you'd like to use for the entire validation, or if you are using instrumental data
you've created yourself, comment-in the line: "instrument = 'GOES-15'"

FORECAST TIME BEFORE EVENT
---
FORMAT: integer  
EX: 1  
OTHER NOTES: the amount of days before the event that a model may have predictions for.
For example, if your model can only predict an event one day before it happens, 
put in 1. If your model has the potential to predict an event 5 days before it
happens, put in 5.

START AND END DATES
---
FORMAT: integers  
EX: 2010, 5, 1  
OTHER NOTES: you do not actually have to have model output for your beginning and
end dates. ValRE will simply extract all model files that are dated within the date
range that you have specified. If you do not have dates specified in the names
of your model files, ValRE will NOT be able to function!

THRESHOLDS
---
FORMAT: lists of integers or floats  
EX: [10,100]  
OTHER NOTES: indices of threshold lists must correspond to each other. Ie, if the first
value in energy_thresholds is 10, the first value of the pfu_thresholds must be the
corresponding flux value of 10. If the model you are using is probabilistic, you may
comment out pfu_threshold; similarly, if the model you are using is not probabilistic,
you may comment out prob_threshold. If prob_threshold is commented out for a
probabilistic model, ValRE will assume that the model output files contain probability
thresholds and will read them in instead.
Finally, ValRE will check if a model does not have output for each threshold and
if it doesn't simply won't create a report for that threshold.

REPORTS
---
FORMAT: True or False boolean  
OTHER NOTES: this is not either/or, both can be marked True if you'd like both reports
and both can be marked False if you'd like none.

MANUAL ADDITIONS
---
FORMAT: lists of integers  
EX. [0,5]  
OTHER NOTES: All of these value should be 0 if you have not completed any calculations yourself. Lists must be the same length as 
energy_threshold determined in the thresholds category above, and each value is the amount of manual hits, misses, correct negatives, or 
false alarms for the corresponding energy_threshold that has been calculated manually. These values can be added to if there are 
particular events that ValRE is not able to use model and/or observational output to calculate, but you would still like to be included 
for calculations of metric scores.

NO MODEL FILE
---
FORMAT: string  
OPTIONS: 'all_clear' or 'nothing'  
OTHER NOTES: if no_mod_file = 'all_clear', not having model files for a given event will be read as equivalent to a forecast all clear.
If no_mod_file = 'nothing', ValRE will skip all observation files that don't have corresponding model files.
