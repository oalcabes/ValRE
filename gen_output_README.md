# gen_output documentation

gen_output.py is a module intended to be used to generate observation JSON files by extracting GOES data.
Below find documentation of all of the functions gen_output offers.

## obs_csv2json(input_file,output_file,example_path,instrument)
### a function converting csv output files from operational_sep_quantities to json files for observations
**input_file: *str***  
  csv filename (must end in .csv)  
**output_file: *str***  
  json filename (must end in .json)  
**example_path: *str***  
  path where template json file can be found  
**instrument: *str***  
  name of the instrument used for observation  
  
## choose_inst(given_start_date,given_end_date)
### choose the correct instrument to use for observations for a given date range. Used if there is no information about which instrument was primary.
**given_start_date: *date object***  
  start date of event  
**given_end_date: *date object***  
  end date of event  
  
## choose_prime_inst(given_start_date,given_end_date)
### choose the correct instrument to use for observations for a given date range based on the primary instrument for that time period. inputs must be date objects from the datetime module.
**given_start_date: *date object***  
  start date of event  
**given_end_date: *date object***  
  end date of event  

## database_extraction(mod_start_time,mod_end_time,instrument_chosen,subevent_bool)
### a function that creates observational json output files given start and end dates by extracting data from the GOES database. Only works with GOES instruments.
**mod_start_time: *datetime object***  
  start time of model prediction window  
**mod_end_time: *datetime object***  
  end time of model prediction window  
**instrument_chosen: *boolean***  
  boolean of whether or not an instrument has been selected for this event yet. True if an instrument has been selected, False if not.  
**subevent_bool: *boolean***  
  boolean of whether or not the event is a subevent, ie, has crossed the thresholds of >10 MeV >10 pfu or >100 MeV >1 pfu.  

## gen_subevent_bools(p_10,p1_100)
### given lists of peak fluxes for protons >10 MeV and >100 MeV, creates a boolean for whether or not each event is a subevent (doesn't cross a threshold)
**p_10: *list of floats***  
  list of peak intensities of protons with >10 MeV
**p_100: *list of floats***  
  list of peak intensities of protons with >100 MeV  

## multi_events(start_time,end_time,subevent_bools)
**start_time: *list of datetime objects***  
  list of start times of events  
**end_time: *list of datetime objects***  
  list of end times of events  
**subevent_bools: *list of booleans***  
  list of booleans of whether or not the event is a subevent 
