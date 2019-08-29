
ValRE EDITS
===========

THIS DOCUMENT HAS NOTES FOR ANYONE WHO MAY HAVE TO ALTER THE ValRE CODE. READ BELOW IF ValRE
IS NOT WORKING CORRECTLY AND YOU NEED TO EDIT IT.

what ValRE does:
- reads in names of all available observation output files
- extracts files between given start and end dates
- extracts values from each observation file and determines observed all clear value
- looks through all model files and extracts any files that correspond to this observed event
- determines forecast all clear based on if any of the corresponding model files say a threshold will be crossed
- compares observed and forecasted all clear to generate metrics
- uses other values (ex. peak intensity, forecast probabilities, etc.) to create verification graphs
- prints reports in PDF and/or JSON formats
  - note: plots are included in the PDF report, but not in the JSON report. Individual PNG files of plots can be found in the
  validation_reports/figures folder.


Likely Problems with ValRE
==========================
1. NOT LOADING IN FILES CORRECTLY: There are a few things you may need to change in the code if it isn't loading in your files 
correctly, and you can't just change your output files.  
a. INPUT FILES: to load in your input files, you may need to change the format of the datestring that is in the file name. If
   it is necessary to change this in the actual code as opposed to just changing your files, you can edit the date_range function
   lines:  
   
   ```python
      d2_str1 = str(d2)
      d2_str2 = d2.strftime('%Y_%m_%d')
      for f in all_files:
         if d2_str1 in str(f) or d2_str2 in str(f):
   ```
   If you need to add a new date format, simply create
   ```python
   d2_str3 = d2.strftime('your new format')
   ```
   and append the line
   ```python
   if d2_str1 in str(f) or d2_str2 in str(f):
   ```
   to
   ```python
   if d2_str1 in str(f) or d2_str2 in str(f) or d3_str3 in str(f):
   ```
   b. OUTPUT FILES: Similarly to input files, if you're having problems loading your output files and need to change the form of the
   datestring, you can alter the following lines in the main function of the code:
   ```python
   if str(mod_start_time.date()) in str(obs_files) or mod_start_time.strftime('%Y_%m_%d') in str(obs_files):
      for obs_f in obs_files:
          if str(mod_start_time.date()) in str(obs_f) or mod_start_time.strftime('%Y_%m_%d') in str(obs_f):
   ```
   and again, you can just add another or statement in both if statements with:
   ```python
   mod_start_time.strftime(your format) in str(obs_f)
   ```          
2. MODEL PREDICTION WINDOW: some models do not have predictions windows, but simply have one value at one time (for example, calculating
a flux based off of a CME speed). If you have such a model and need to run it through ValRE, I suggest pre-processing your model start
and end times according to the times the model would actually be predicting for. If more alterations are necessary, edits to ValRE would 
likely occur in the following if statement in the main code:
```python
if mod_end_time < obs_start_time:
    #model forecast ends before event starts
    print('model file too early - skipping')            
elif obs_start_time < mod_start_time:
    #model forecast begins after event ends
    print('model file too late - skipping')                
elif obs_start_time < mod_end_time:
    #model forecast starts before event begins and ends after
    #event starts, so using this file
    print('model file recorded event')
```
          
          
Likely Problems with gen_output Module
======================================
1. UPDATES TO operational_sep_quantities.py: If there is a new version of operational_sep_quantities.py, you can replace the old copy of
the file in the ValRE folder with the new copy. However, if the inputs or outputs to the function have changed, it may be necessary to
alter the way gen_ouput.py incorporates the code accordingly.  
    a. CHANGES TO INPUT: Most alterations to the input can be made in the database_extraction function in the gen_output.py code, before
    the sep.run_all command is run.  
    b. CHANGES TO OUTPUT: The current output files of operational_sep_quantities.py (as of July 2019) are csv documents with various 
    important values. The function (located in the ValRE folder) output_to_json.py includes a function called obs_csv2json, which loads 
    in all of the values from the csv files and reformats them into the JSON format given by the CCMC. If the output of 
    operational_sep_quantities.py changes or has problems, these observational JSON files may be created incorrectly. Therefore, it may 
    be necessary to edit the obs_csv2json function directly to make sure all of the values are correctly loaded into JSON.  
2. DETECT PREVIOUS EVENT VALUE: This is an input value for the operational_sep_quantities.py module. For most ValRE purposes, the detect
previous event value can be set to false, as most events that ValRE uses do not occur right after each other. However, for such an
event, if it is necessary to go into the GOES database to retrieve observational data, it may be necessary to change the detect previous
event value to True. ValRE does not currently have a way to automate this, and it may be necessary to validate that particular event
manually. If you need to add in an automation, you can edit the function database_extraction in gen_output.py and change default input to your own value.
