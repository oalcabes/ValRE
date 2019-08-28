# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:45:12 2019

@author: oalcabes
"""

#generating observation output code

from datetime import timedelta, datetime
import operational_sep_quantities as sep
import operational_sep_quantities_one as one_sep
from importlib import reload
from dateutil.parser import parse
from urllib import request
import config as cfg
import pandas as pd
from pathlib import Path
import numpy as np
import csv
import json as js
import os.path
import argparse

#%%

#some paths
#directory containing json template
ref_path = Path('ref_files')
#directory where csv files are created
katies_path = Path('output')

#booleans
instrument_chosen = False
detect_previous_event = False #this is really whats left
#%%

### FUNCTIONS ###

def obs_csv2json(input_file,output_file,example_path,instrument):
    """
    a function converting csv output files from operational_sep_quantities to json
    files for observations
    """

    obs_path = Path(cfg.obs_path)
    
    with open(example_path,'r') as e:
        example = js.load(e)
    
    #deleting unused categories
    del(example['sep_forecast_submission']['forecasts'])
    del(example['sep_forecast_submission']['triggers'][2])
    del(example['sep_forecast_submission']['triggers'][1])
    del(example['sep_forecast_submission']['triggers'][0])
    del(example['sep_forecast_submission']['triggers'][0]['particle_intensity']['instrument'])
    del(example['sep_forecast_submission']['triggers'][0]['particle_intensity']['last_data_time'])
    del(example['sep_forecast_submission']['contacts'])
    del(example['sep_forecast_submission']['model'])
    del(example['sep_forecast_submission']['issue_time'])
    
    example['sep_forecast_submission']['mode'] = 'observation'

    #json template for observations
    obs_json = example

    fieldnames = ('energy_threshold','flux_threshold','start_time','intensity',
                  'peak_time','rise_time','end_time','duration','fluence>10',
                  'fluence>100')

    #extracting data from csv file
    with open(input_file,'r') as f:
        reader = csv.DictReader(f, fieldnames)
        out = js.dumps( [ row for row in reader ] )

    obs_data = js.loads(out)

    data={}
    (obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']
             ['observatory']) = instrument

    #creating data for all energy levels forecast
    for j in range(1,len(obs_data)):
        data[j-1]=obs_data[j]

    #recording start and end times for all events
    for i in range(len(data)):
        data[i]['start_time'] = datetime.strptime(data[i]['start_time'],'%Y-%m-%d %H:%M:%S')
        data[i]['start_time'] = data[i]['start_time'].isoformat()
        data[i]['end_time'] = datetime.strptime(data[i]['end_time'],'%Y-%m-%d %H:%M:%S')
        data[i]['end_time'] = data[i]['end_time'].isoformat()
        data[i]['peak_time'] = datetime.strptime(data[i]['peak_time'],'%Y-%m-%d %H:%M:%S')
        data[i]['peak_time'] = data[i]['peak_time'].isoformat()
    
        #recording observed values for all events
        if i > 0:
            (obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']
                     ['ongoing_events']).append({})

        event = (obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']
                     ['ongoing_events'][i])
        
        #start and end times
        event['start_time']=data[i]['start_time']
        event['threshold'] = data[i]['flux_threshold']
        event['energy_min'] = float(data[i]['energy_threshold'][1:])
        event['energy_max'] = -1
        event['end_time']=data[i]['end_time']

        #peak values
        event['peak_intensity']=data[i]['intensity']
        event['peak_time'] = data[i]['peak_time']
        event['intensity_units']='pfu'
            
        #fluence values
        event['fluence'] = [{'energy_min' : '10','fluence_value' : 'fluence_value',
                             'units' : 'MeV [cm^-2]'},
                            {'energy_min' : '100', 'fluence_value' : 'fluence_value',
                             'units' : 'MeV [cm^-2]'}]
        event['fluence'][0]['fluence']=data[i]['fluence>10']
        event['fluence'][1]['fluence']=data[i]['fluence>100']


        if float(event['peak_intensity']) >= cfg.pfu_threshold[cfg.energy_threshold.index(int(event['energy_min']))]:
            event['all_clear_boolean'] = 'false'

        else:
            event['all_clear_boolean'] = 'true'


    #building json file
    with open(obs_path / output_file, 'w') as s:
        js.dump(obs_json,s,indent=1)
        print('json file %s created' %output_file)
     
    return

def choose_inst(given_start_date,given_end_date): #INPUTS MUST BE DATE OBJECTS
    """
    choose the correct instrument to use for observations for a given date
    range. inputs must be date objects from the datetime module. used if there
    is no information about which instrument was primary.
    """

    inst_start_dates=[]
    inst_end_dates=[]
    good_instruments = []
    good_end_dates = []
    bad_inst = []

    #extracting dates where instruments are active from csv file
    inst_dates = pd.read_csv(ref_path / 'instrument_dates.csv')

    for s in inst_dates['start']:
        inst_start_dates.append(datetime.strptime(str(s),'%Y-%m').date())

    for e in inst_dates['end']:
        if str(e) == 'nan':
            inst_end_dates.append(datetime.today().date())
        else:
            inst_end_dates.append(datetime.strptime(str(e),'%Y-%m').date())

    #checking which instruments are active during given time period and
    #choosing the correct ones
    print('checking which instruments are active for given dates')

    for i in range(len(inst_start_dates)):
        if (inst_start_dates[i] < given_start_date) and (given_end_date <
           inst_end_dates[i]):
            print('%s works' %inst_dates['Instrument'][i])
            good_instruments.append(inst_dates['Instrument'][i])
            good_end_dates.append(inst_end_dates[i])
        else:
            print('outside of %s range' %inst_dates['Instrument'][i])

    #checking if active instruments actually have data for that date
    for inst in good_instruments:
        inst_str = inst.replace('-','').lower()
        year = str(given_start_date).split('-')[0]
        month = str(given_start_date).split('-')[1]
        url = ('https://satdat.ngdc.noaa.gov/sem/goes/data/avg/'+ year + '/' +
               month + '/' + inst_str)

        try:
            request.urlopen(url)
            print('%s data available' %inst)

        except:
            print('%s data NOT available' %inst)
            bad_inst.append(inst)

    #not choosing instrument if it doesn't have data
    for binst in bad_inst:
        good_instruments.remove(binst)

    #if more than one instrument is available, choose which one to use
    if len(good_instruments) > 1:
        print('Please choose which instrument you would like to use.')

        #ADD IN SOMETHING ABOUT PRIMARY INSTRUMENT HERE

        for j in range(len(good_instruments)):
            print('Type ' + str(j) + ' for ' + str(good_instruments[j]))

        inst_choice = input('Answer:' )

        instrument = good_instruments[int(inst_choice)]
        end_date = good_end_dates[int(inst_choice)]

        print('we are using %s as our instrument for observations' %instrument)

    else:

        instrument = good_instruments[0]
        end_date = good_end_dates[0]
        print('we are using %s as our instrument for observations' %instrument)

    return([instrument,end_date])


def choose_prime_inst(given_start_date,given_end_date):
    """
    choose the correct instrument to use for observations for a given date
    range based on the primary instrument for that time period. inputs must be
    date objects from the datetime module.
    """

    #extracting primary dates where instruments are active from csv file
    inst_prime_dates = pd.read_csv(ref_path / 'GOES_primary_assignments.csv', header=3)

    #prime instrument option
    for d in range(len(inst_prime_dates['Start Date'])):
        change_date = parse(inst_prime_dates['Start Date'][d])
        if given_start_date >= change_date.date():
            prime_inst = inst_prime_dates['EPEAD Primary'][d]
            backup_inst = inst_prime_dates['EPEAD Secondary'][d]
            end_date = parse(inst_prime_dates['Start Date'][d+1]).date()

            if str(prime_inst) == 'nan':
                print('no information about primary instrument available.'
                      'Choosing instrument based on active date ranges')
                alternate_output = choose_inst(given_start_date,given_end_date)

                return(alternate_output)

            break

    prime_inst = str(prime_inst).split('.')[0]

    if len(prime_inst) == 2:
        inst_str = str(prime_inst)
    elif len(prime_inst) == 1:
        inst_str = '0' + str(prime_inst)

    print('GOES-%s is the primary instrument for given start time' %inst_str)

    year = str(given_start_date).split('-')[0]
    month = str(given_start_date).split('-')[1]
    url = ('https://satdat.ngdc.noaa.gov/sem/goes/data/avg/'+ year + '/' +
           month + '/goes' + inst_str)

    try:
        request.urlopen(url)
        print('GOES-%s has data available' %inst_str)
        instrument = 'GOES-' + inst_str
        print('we are using %s as our instrument for observations' %instrument)

    except request.HTTPError:
        print('GOES-%s does NOT have data available' %inst_str)

        if len(str(backup_inst)) == 2:
            inst_str = str(backup_inst)
        elif len(str(backup_inst)) ==1:
            inst_str = '0' + str(backup_inst)

        print('checking for data from backup instrument GOES-%s' %inst_str)

        url = ('https://satdat.ngdc.noaa.gov/sem/goes/data/avg/'+ year + '/'
               + month + '/goes' + inst_str)

        try:
            request.urlopen(url)
            print('backup instrument data found - using backup instrument')
            instrument = 'GOES-' + inst_str
            print('we are using %s as our instrument for observations'
                  %instrument)

        except request.HTTPError:
            print('no knowledge of backup or primary instrument - choosing '
                  'instrument based on available data')
            alternate_output = choose_inst(given_start_date,given_end_date)

            return(alternate_output)

    return([instrument,end_date])

def database_extraction(mod_start_time,mod_end_time,instrument_chosen,subevent_bool,
                        detect_previous_event = False,thresholds='100,1',one_thresh = False):
    
    #NOTE: NEED TO FIX THRESHOLDS STUFF WITH THIS RN
    
    """
    a function that creates observational json output files given start and end dates
    by extracting data from the iswep GOES database. Only works with GOES instruments.
    """
    obs_file_created = False

    #extending time window
    window_end_time = (mod_end_time + timedelta(days=2))
    window_start_time = (mod_start_time - timedelta(days=2))
    
    #making a list of all dates within window
    day_list=[]
    for d in range(10):
        day_list.append((window_start_time + timedelta(days=d)).date())
    print('day list = %s' %day_list)
    
    print('determining if an instrument has been chosen')

    if instrument_chosen:
        #if an instrument has been chosen, checking to make sure it still works for this date
        if inst_end < window_end_time:
            instrument_chosen = False
    else:
        #if insturment hasn't been chosen, figuring out what it should be for given date
        try:
            #if instrument is specified in cfg using that
            instrument = cfg.instrument
            inst_end = datetime.today()
            print('using %s as our instrument for observations' %instrument)
            instrument_chosen = True

        except:
            #choosing instrument using function if not given in cfg
            instrument_stuff = choose_prime_inst(window_start_time.date(),
                                                 window_end_time.date())
            instrument = instrument_stuff[0]
            #figuring out how long we can use this instrument
            inst_end = instrument_stuff[1]
            instrument_chosen = True
    
    #running katie's code to extract data using chosen instrument and dates
    print('extracting data from GOES website')
    
    if one_thresh:
        one_sep.run_all(str(window_start_time), str(window_end_time), str(instrument),
                        'integral', '', '', True, detect_previous_event, thresholds)    
        print('ran for threshold %s' %thresholds)
    else:
        if subevent_bool:
            thresholds = '10,1'
            #if event is a subevent, changing the threshold in katie's code to
            #10 MeV > 1pfu so that it will be recorded
            print('********************SUBEVENT**************************')
            sep.run_all(str(window_start_time), str(window_end_time), str(instrument),
                        'integral', '', '', True, detect_previous_event, thresholds)
            print('ran for subevent')
        else:
            #if an event, running with usual thresholds
            print('********************EVENT*****************************')
            sep.run_all(str(window_start_time), str(window_end_time),str(instrument), 
                        'integral', '', '', True, detect_previous_event, thresholds)
        
    #reloading function so it doesn't keep old data    
    reload(sep)
    
    #reformatting csv created from katie's code to json
    print('extracted - reformatting')        
    for day in day_list:    
        if not obs_file_created:
            #checking each day within the window to find the csv file if it hasn't
            #already been found
            print('thresholds: %s' %thresholds)
               
            if one_thresh:
                new_obs_name = ('sep_values_' + str(instrument) + '_integral_gt' +
                                str(thresholds).split(',')[0] + '_' + str(thresholds).split(',')[1] + 'pfu_' +
                                day.strftime('%Y_%m_%d').replace('_0','_') + '.csv')
            else:
                new_obs_name = ('sep_values_' + str(instrument) + '_integral_' +
                                day.strftime('%Y_%m_%d').replace('_0','_') + '.csv')
                
            print('new_os_name %s' %new_obs_name)        
            
            #checking if that file exists
            if os.path.exists(katies_path / new_obs_name):
                #if a file with this date exists, creating the corresponding json file
                
                #json name
                if one_thresh:
                    obs_name = (str(instrument) + '_' + str(day) + 'only_' + str(thresholds).split(',')[0] + 'MeV_event.json')
                else:
                    obs_name = (str(instrument) + '_' +
                                str(day) + '.json')
                #creating json file
                obs_csv2json((katies_path / new_obs_name), obs_name,
                             (ref_path/'example_sepscoreboard_json_file_v20190228.json'),
                             instrument)
            
                print('obs file created')
                #file is created - will not run for anymore dates within window
                obs_file_created = True
                
                return(obs_name)
            else:
                print('no csv file found with this date, checking next one')
        #if the json file has been created, not running for anymore dates
        #else:
            #break
            
        
def two_in_one(obs_file,et,subevent):
    """
    will create JSON output files if there are two events (for each threshold) in one
    time window. Ie, if there are two >10MeV >10pfu events as well as two >100MeV >1pfu
    events, will create files for all four events, but if there are three >100MeV >1pfu
    events, will only generate JSON files for the first two. Second events have different
    thresholds in different files as opposed to together.
    """
    
    #in this function, the "original time window" talked about in the comments
    #refers to the start and end times that were input to create the file obs_file,
    #which will likely have been created using the database_extraction function
    
    #opening first output file created by operational_sep_quantities
    with open(obs_file, 'r') as o:
        out = js.load(o)
    
    #all events recorded in that output file
    ongoing_events = (out['sep_forecast_submission']['triggers'][0]['particle_intensity']
                          ['ongoing_events'])
    
    #creating lists for values from each event
    end_times = []                                                                                                                                            
    start_times = []
    energy_thresholds = []
    flux_thresholds = []
    out_names = []
    
    #appending values to lists for each event
    for i in range(len(ongoing_events)):
        start_times.append(parse(ongoing_events[i]['start_time']))
        end_times.append(parse(ongoing_events[i]['end_time']))
        energy_thresholds.append(ongoing_events[i]['energy_min'])
        flux_thresholds.append(float(ongoing_events[i]['threshold']))
    
    #checking if there was a second event for each threshold
    for i in range(len(end_times)):
        end = end_times[i]
        #if the end time of an event for any threshold was a day before the last day
        #in the original time window given, will check if ONLY THAT THRESHOLD
        #had another event after the first one, using the end time of the first
        #event of that threshold as the new start time of the event window
        if end.date() < et.date():
            print('end time to use as new start time: %s' %end)
            #figuring out which threshold this end time was for
            flux_thresh = int(flux_thresholds[i])
            energy_thresh = int(energy_thresholds[i])
            print('extracting second event for threshold ' + str(flux_thresh) + ' MeV '
                  + str(energy_thresh) + ' pfu')
            #new start time (2 days in advance bc the database_extraction function
            #makes the start time 2 days prior, so will cancel that out)
            st = end + timedelta(days=2)
            #thresholds in correct format
            thresholds = str(energy_thresh) + ',' + str(flux_thresh)
            print('thresholds: %s' %thresholds)
            #creating observation data for second event for thresholds given
            out_names.append(Path(cfg.obs_path) /
                             database_extraction(st,et,instrument_chosen,subevent,
                                                 thresholds = thresholds,
                                                 one_thresh = True))
            
    #returns list of all new files created by this function
    return(out_names)
    
        
def multi_event(st,et,instrument_chosen,subevent):
    """
    all events in one time window (not just two)
    
    used if there is more than one event occurring within a short time period. will
    generate an output file for every event that occurs within a given time window -
    not to be confused with many_events, which generates output given multiple time
    windows. Can create files for up to 3 events within specified time window.
    """
    print('checking for multiple events within given time window')
    
    #creating file for time window with first events for all thresholds
    out_name = Path(cfg.obs_path) / database_extraction(st,et,instrument_chosen,subevent)

    #creating files for all second events for all thresholds
    new_files = two_in_one(out_name,et,subevent)
    
    #creating files for any third events for all thresholds that had a second event
    for file in new_files:
        two_in_one(file,et,subevent)        
    
    return
        
def gen_subevent_bools(p_10,p_100):
    """
    given lists of peak fluxes for protons >10 MeV and >100 MeV, creates a boolean
    for whether or not each event is a subevent (doesn't cross a threshold)
    """
    #list of subevent booleans
    subevent_bools = []
    
    #extracting 10 MeV peak flux if it exists
    for j in range(len(p_10)):
        try:
            p10 = float(p_10[j])
        except ValueError:
            p10 = 'nan'
        
        #extracting 100 MeV peak flux if it exists
        try:
            p100 = float(p_100[j])
        except ValueError:
            p100 = 'nan'
        
        #checking if peak fluxes exist
        if str(p10) != 'nan' and str(p100) != 'nan':
            #if the peak fluxes both exist and >10 MeV is both below threshold,
            #subevent is true (only care about >10 bc of definition of subevent)
            if p10 < 10:
                subevent_bools.append(True)
            elif p10 > 10:
                subevent_bools.append(False)
        
        #if >10 MeV doesn't exist, subevent is true
        else:
            subevent_bools.append(True)
            
    return(subevent_bools)
        
def many_events(start_time,end_time,subevent_bools):
    """
    takes in lists of start times and end times to create a list of time windows,
    and a list of whether or not an event is a subevent, and uses those lists to run
    functions that extract data from the GOES database. Each list must have
    the same length, and indices of lists must correspond (ie start_time[j] has an end
    time of end_time[j] and its subevent boolean is subevent_bools[j]). not to be
    confused with multi_events, which generates output given multiple events within one
    time window.
    """
    
    #running through for each event
    for j in range(len(start_time)):
        
        #start, end, and subevent bool for this event
        st = start_time[j]
        et = end_time[j]
        subevent = bool(subevent_bools[j])
        
        #checking if start time is actually available
        if str(st) != 'nan':
            try:
                st = parse(st)
                yes_st = True
            except ValueError:
                yes_st = False
        else:
            yes_st = False
        
        #checking if end time is actually available
        if str(et) != 'nan':
            try:
                et = parse(et)
                yes_et = True
            except ValueError:
                yes_et = False
        else:
            yes_et = False
    
        #if both start and end times are available, running the code
        if yes_st and yes_et:
            #event must be after Nov. 2010 because currently no capability for
            #instruments in use before then
            if st > datetime(2010,9,1):
                try:
                    print('got start and end times! running database extraction')  
                    database_extraction(st,et,instrument_chosen,subevent)
                except:
                    continue
            else:
                print('cannot run for events before November 2010 because do not have '
                      'access to instruments before then')
        