# ValRE

Readme file for ValRE radiation event validation software written in Python 3.

NOTE: IF YOU NEED TO ALTER THE CODE, PLEASE READ ValRE_ADDITIONS.TXT

Purpose
========
ValRE validates spaceweather forecasting models of SEP events. Validation is done by
comparing model forecasts of events to historical observations and generating skill scores,
which are written into reports (in PDF and/or JSON format). ValRE also creates plots
for qualitative verification, which appear in the PDF and are present as PNG files in the
validation_reports/figures folder.

***NOTE: ValRE was not intended for the creation of model output. Model output must already
be present before running ValRE; however, given model output, ValRE has the capability of
generating the necessary observational output from the iswa GOES database.

Included With the ValRE Package:
================================
ValRE_README.txt
ValRE_ADDITIONS.txt
ValRE.py
config.py
operational_sep_quantities.py
output_to_json.py
validation_reports (inner directory - may not be present until first run)
validation_reports/figures (inner directory that will house PNG files)
GOES_primary_assignments.csv
instrument_dates.csv

Python Libraries
================
before running ValRE, you must make sure you have the correct Python libraries loaded on your computer.
Most of these libraries are included in Anaconda Navigator. However, it may be necessary to install some
yourself. In order to do so, type: "pip install <library>".

This program makes use of the software "operational_sep_quantities" created by Katie Whitman, a program
which has extensive that can be found with the command "pydoc operational_sep_quantities".

The full list of modules that ValRE uses can be found at the beginning of the ValRE.py code.

Usage instructions
==================
Note about model output: currently, a date in the format YYYY-MM-DD or YYYY_MM_DD MUST be included in the name of
each model output file and each observation output file (if you already have observation data). If this
is a problem, please read ValRE_ADDITIONS.txt.

Once you are sure you have correct model output, open config.py in your python editor of choice -
Vi, Vim, or Spyder are just a few examples of options if you don't already have one.

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

The formats in which all of these values must be given in are detailed in the comments of the
configuration file, as well as in the "input formats" section below.

Once you've filled out the configuration file, you can run ValRE from the command line. Once ValRE is finished
running, you can find your reports in the validation_reports directory, and PNGs of your figures in the figures
validation_reports/figures directory.

Input Formats
=============
PATHFILES
FORMAT: output directories in the form of a string
EX. '.\Mag4_output'
OTHER NOTES: Write the directory however you would on your own computer.
The current working directory is the directory you have placed the ValRE package in.
If you do not have dates specified in the names
of your model files, ValRE will NOT be able to function

---
MODEL NAME
FORMAT: string
EX. 'MAG4'
OTHER NOTES: this will only be used for report-writing purposes

---
INSTRUMENT FOR OBSERVATIONS
FORMAT: string
EX: 'GOES-08'
OPTIONS: GOES-08, -10, -11, -12, -13, -14 -15, SEPEM
OTHER NOTES: ValRE automatically chooses the correct GOES observational instrument to
use for your given start and end dates. However, if you'd like to specify one instrument
you'd like to use for the entire validation, or if you are using instrumental data
you've created yourself, comment-in the line below:
instrument = 'GOES-15''

---
START AND END DATES
FORMAT: integers
EX: 2010, 5, 1
OTHER NOTES: you do not actually have to have model output for your beginning and
end dates. ValRE will simply extract all model files that are dated within the date
range that you have specified. If you do not have dates specified in the names
of your model files, ValRE will NOT be able to function!

---
DETECT PREVIOUS EVENT
write doc here ******* / possibly add this into code and just document it in the editing
documentation file

---
THRESHOLDS
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

---
REPORTS 
FORMAT: True or False boolean
OTHER NOTES: this is not either/or, both can be marked True if you'd like both reports
and both can be marked False if you'd like none.

More extensive documentation
============================
for more extensive documentation, ValRE utilizes the pydoc module. Find the documentation by typing
"pydoc ValRE". Please note that doing so will also run the program. 
