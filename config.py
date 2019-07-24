# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:37:11 2019

@author: oalcabes
"""

#CONFIG FILE - FOR WHEN MODEL OUTPUT HAS ALREADY BEEN CREATED

### PATHFILES ###
#FORMAT: output directories in the form of a string
#EX. '.\Mag4_output'
#OTHER NOTES: Write the directory however you would on your own computer.
#The current working directory is the directory you have placed the ValRE package in.
#If you do not have dates specified in the names
#of your model files, ValRE will NOT be able to function!

#model output path:
model_path = '..\SEPSTER_output/'
#observational output path:
obs_path = '..\GOES_output/'


### MODEL NAME ###
#FORMAT: string
#EX. 'MAG4'
#OTHER NOTES: this will only be used for report-writing purposes

model_name = 'SEPSTER'


### INSTRUMENT FOR OBSERVATIONS ###
#FORMAT: string
#EX: 'GOES-08'
#OPTIONS: GOES-08, -10, -11, -12, -13, -14 -15, SEPEM
#OTHER NOTES: ValRE automatically chooses the correct GOES observational instrument to
#use for your given start and end dates. However, if you'd like to specify one instrument
#you'd like to use for the entire validation, or if you are using instrumental data
#you've created yourself, comment-in the line below:
#instrument = 'GOES-15'


### START AND END DATES ###
#FORMAT: integers
#EX: 2010, 5, 1
#OTHER NOTES: you do not actually have to have model output for your beginning and
#end dates. ValRE will simply extract all model files that are dated within the date
#range that you have specified. If you do not have dates specified in the names
#of your model files, ValRE will NOT be able to function!

#set start date
start_year = 2010
start_month = 5
start_day = 1

#set end date
end_year = 2019
end_month = 6
end_day = 2


### :) ###
#write doc here ******* / possibly add this into code and just document it in the editing
#documentation file
detect_previous_event = False


### THRESHOLDS ###
#FORMAT: lists of integers or floats
#EX: [10,100]
#OTHER NOTES: indices of threshold lists must correspond to each other. Ie, if the first
#value in energy_thresholds is 10, the first value of the pfu_thresholds must be the
#corresponding flux value of 10. If the model you are using is probabilistic, you may
#comment out pfu_threshold; similarly, if the model you are using is not probabilistic,
#you may comment out prob_threshold. If prob_threshold is commented out for a
#probabilistic model, ValRE will assume that the model output files contain probability
#thresholds and will read them in instead.
#Finally, ValRE will check if a model does not have output for each threshold and
#if it doesn't simply won't create a report for that threshold.

#threshold(s) (in MeV)
energy_threshold = [10,100]
#threshold(s) (in pfu)
pfu_threshold = [10,1]
#probabilisitic threshold (can be just one value for all) ***DECIDE ABT THIS TOO
prob_threshold = [0.25] #consider this


### REPORTS ###
#FORMAT: True or False boolean
#OTHER NOTES: this is not either/or, both can be marked True if you'd like both reports
#and both can be marked False if you'd like none.

#would you like a version of the report in PDF form? True if yes, False if no
PDF_report = True

#would you like a version of the report in JSON form? True if yes, False if no
JSON_report = True