# gen_output documentation

gen_output.py is a module intended to be used to generate observation JSON files by extracting GOES data.
Below find documentation of all of the functions gen_output offers.

## obs_csv2json(input_file,output_file,example_path,instrument)
### a function converting csv output files from operational_sep_quantities to json files for observations
**PARAMETERS**  
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
**PARAMETERS**  
**given_start_date: *date object***  
  start date of event  
**given_end_date: *date object***  
  end date of event  
  
**RETURNS**  
  list containing string of instrument name and datetime object of final date that instrument is available  
  
## choose_prime_inst(given_start_date,given_end_date)
### choose the correct instrument to use for observations for a given date range based on the primary instrument for that time period. inputs must be date objects from the datetime module.
**PARAMETERS**  
**given_start_date: *date object***  
  start date of event  
**given_end_date: *date object***  
  end date of event  
  
**RETURNS**  
  list containing string of instrument name and datetime object of final date that instrument is available  

## database_extraction(mod_start_time,mod_end_time,instrument_chosen,subevent_bool,detect_previous_event=False,thresholds='100,1',one_thresh=False)
### a function that creates observational json output files given start and end dates by extracting data from the GOES database. Only works with GOES instruments.
**PARAMETERS**  
**mod_start_time: *datetime object***  
  start time of model prediction window  
**mod_end_time: *datetime object***  
  end time of model prediction window  
**instrument_chosen: *boolean***  
  boolean of whether or not an instrument has been selected for this event yet. True if an instrument has been selected, False if not.  
**subevent_bool: *boolean***  
  boolean of whether or not the event is a subevent, ie, has crossed the thresholds of >10 MeV >10 pfu or >100 MeV >1 pfu.  
**detect_previous_event: *boolean***
  boolean of whether or not to detect if an event is already happening; that is, if the flux is already above threshold at the beginning
  of the given time window. Default is False. Should only be set to true when running for multiple events that are close to each other,
  likely using the two_in_one or multi_event functions.
**thresholds: *string***
  string of which threshold to run operational_sep_quantities or operational_sep_quantities_one for. More details found in 
  operational_sep_quantities documentation. Default is '100,1'.
**one_thresh: *boolean***
  boolean of whether or not to run operational_sep_quantities for multiple thresholds or operational_sep_quantities_one for one       \
  threshold. Default is False.  
  
**RETURNS**
  name of json file created (string)
  
## two_in_one(obs_file,et,subevent)
### two events in one time window
**PARAMETERS**
**obs_file: *path object***  
  an observational file that may not contain all of the events that ocurred within its given time window  
**et: *datetime object***  
  the end time of the time window originally used when creating the obs_file  
**subevent: *boolean***  
  boolean of whether or not the event is a subevent, ie, has crossed the thresholds of >10 MeV >10 pfu or >100 MeV >1 pfu.  
  
**RETURNS**  
  list of names of all files created (strings)
  
## multi_event(st,et,instrument_chosen,subevent)
### all events in one time window (not just two). Used if there is more than one event occurring within a short time period. will generate an output file for every event that occurs within a given time window - not to be confused with many_events, which generates output given multiple time windows. Can create files for up to 3 events within specified time window.
**PARAMETERS**  
**st: *datetime object***  
  start time of an event window  
**et: *datetime object***  
  end time of an event window  
**instrument_chosen: *boolean***  
  boolean of whether or not an instrument has been chosen for this event  
**subevent: *boolean***  
  boolean of whether or not the event is a subevent, ie, has crossed the thresholds of >10 MeV >10 pfu or >100 MeV >1 pfu.  

## gen_subevent_bools(p_10,p1_100)
### given lists of peak fluxes for protons >10 MeV and >100 MeV, creates a boolean for whether or not each event is a subevent (doesn't cross a threshold)
**PARAMETERS**  
**p_10: *list of floats***  
  list of peak intensities of protons with >10 MeV  
**p_100: *list of floats***  
  list of peak intensities of protons with >100 MeV  
  
**RETURNS**
  list of booleans for whether or not an event was a subevent

## many_events(start_time,end_time,subevent_bools)
### takes in lists of start times and end times to create a list of time windows, and a list of whether or not an event is a subevent, and uses those lists to run functions that extract data from the GOES database. Each list must have the same length, and indices of lists must correspond (ie start_time[j] has an end time of end_time[j] and its subevent boolean is subevent_bools[j]). not to be confused with multi_events, which generates output given multiple events within one time window.
**PARAMETERS**  
**start_time: *list of datetime objects***  
  list of start times of events  
**end_time: *list of datetime objects***  
  list of end times of events  
**subevent_bools: *list of booleans***  
  list of booleans of whether or not the event is a subevent 
