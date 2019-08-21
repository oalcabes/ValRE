# gen_output documentation

gen_output.py is a module intended to be used to generate observation JSON files by extracting GOES data from ISWEP.
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
### choose the correct instrument to use for observations for a given date range. Used if there is no information about which
    instrument was primary.
    
**given_start_date**
**given_end_date**

## choose_prime_inst(given_start_date,given_end_date)
**given_start_date**
**given_end_date**

## database_extraction(mod_start_time,mod_end_time,instrument_chosen,subevent_bool)
**mod_start_time**
**mod_end_time**
**instrument_chosen**
**subevent_bool**

## gen_subevent_bools(p_10,p1_100)
**p_10**
**p_100**

## multi_events(start_time,end_time,subevent_bools)
**start_time**
**end_time**
**subevent_bools**
