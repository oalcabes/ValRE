# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:45:12 2019

@author: oalcabes
"""

#generating observation output code

from datetime import timedelta, datetime
import operational_sep_quantities as sep
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
import argparser

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

    #creating fluence report
    fluence = []
    fluence.append({'energy_min' : '10','fluence_value' : 'fluence_value', 'units' : 'MeV [cm^-2]'})
    fluence.append({'energy_min' : '100', 'fluence_value' : 'fluence_value', 'units' : 'MeV [cm^-2]'})

    #creating format for ongoing events
    ongoing_events = { "start_time": "2017-09-10T19:30Z", "threshold": 10,
                       "energy_min": 10, "energy_max": -1 }

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
    for i in range(1,len(obs_data)):
        data[i-1]=obs_data[i]

    #recording start and end times for all events
    for j in range(0,len(data)):
        data[j]['start_time'] = datetime.strptime(data[j]['start_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['start_time'] = data[j]['start_time'].isoformat()
        data[j]['end_time'] = datetime.strptime(data[j]['end_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['end_time'] = data[j]['end_time'].isoformat()
        data[j]['peak_time'] = datetime.strptime(data[j]['peak_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['peak_time'] = data[j]['peak_time'].isoformat()

    #recording observed values for all events
    for i in range(len(data)):
        
        
        if i > 0:
         (obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']
                  ['ongoing_events']).append(ongoing_events)

        event = (obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']
                         ['ongoing_events'][i])
        
        #energy channe;
        event['energy_min'] = float(data[i]['energy_threshold'][1:])
        
        #start and end times
        event['start_time']=data[i]['start_time']
        event['end_time']=data[i]['end_time']

        #peak values
        event['peak_intensity']=data[i]['intensity']
        event['peak_time'] = data[i]['peak_time']
        event['intensity_units']='pfu'
        
        #fluence values
        event['fluence'] = fluence
        event['fluence'][0]['fluence']=data[i]['fluence>10']
        event['fluence'][1]['fluence']=data[i]['fluence>100']

        #thresholds (using default)
        if int(event['energy_min']) == 10:
            event['threshold'] = 10
        elif int(event['energy_min']) == 100:
            event['threshold'] = 1
        
        #calculating all clear based on whether peak intensity crosses threshold
        if float(event['peak_intensity']) >= event['threshold']:
            event['all_clear_boolean'] = 'false'

        else:
            event['all_clear_boolean'] = 'true'

    #building json file
    with open(obs_path / output_file, 'w') as s:
       js.dump(obs_json,s,indent=1)
       print('json file created')
     
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

def database_extraction(mod_start_time,mod_end_time,instrument_chosen,subevent_bool):
    """
    a function that creates observational json output files given start and end dates
    by extracting data from the iswep GOES database. Only works with GOES instruments.
    """
    obs_file_created = False

    #extending time window
    window_end_date = (mod_end_time.date() + timedelta(days=2))
    window_start_date = (mod_start_time.date() - timedelta(days=2))
    
    #making a list of all dates within window
    day_list=[]
    for d in range(10):
        day_list.append(window_start_date + timedelta(days=d))
    print('day list = %s' %day_list)
    
    print('determining if an instrument has been chosen')

    if instrument_chosen:
        #if an instrument has been chosen, checking to make sure it still works for this date
        if inst_end < window_end_date:
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
            instrument_stuff = choose_prime_inst(window_start_date,
                                                 window_end_date)
            instrument = instrument_stuff[0]
            #figuring out how long we can use this instrument
            inst_end = instrument_stuff[1]
            instrument_chosen = True
    
    #running katie's code to extract data using chosen instrument and dates
    print('extracting data from GOES website')
    if subevent_bool:
        #if event is a subevent, changing the threshold in katie's code to
        #10 MeV > 1pfu so that it will be recorded
        print('********************SUBEVENT**************************')
        sep.run_all(str(window_start_date), str(window_end_date), str(instrument),
                    'integral', '', '', False, detect_previous_event, '10,1')
        print('ran for subevent')
    else:
        #if an event, running with usual thresholds
        print('********************EVENT*****************************')
        sep.run_all(str(window_start_date), str(window_end_date),str(instrument), 
                    'integral', '', '', False, detect_previous_event, '100,1')
        
    #reloading function so it doesn't keep old data    
    reload(sep)
    
    #reformatting csv created from katie's code to json
    print('extracted - reformatting')        
    for day in day_list:    
        if not obs_file_created:
            #checking each day within the window to find the csv file if it hasn't
            #already been found
            
            #getting what the csv file name would be for this date
            new_obs_name = ('sep_values_' + str(instrument) + '_integral_' +
                            day.strftime('%Y_%m_%d').replace('_0','_') + '.csv')
            print('new_os_name %s' %new_obs_name)        
            
            #checking if that file exists
            if os.path.exists(katies_path / new_obs_name):
                #if a file with this date exists, creating the corresponding json file
                
                #json name
                obs_name = (str(instrument) + '_' +
                            str(mod_start_time.date()) + '.json')
                #creating json file
                obs_csv2json((katies_path / new_obs_name), obs_name,
                             (ref_path/'example_sepscoreboard_json_file_v20190228.json'),
                             instrument)
            
                print('obs file created')
                #file is created - will not run for anymore dates within window
                obs_file_created = True
                break
            else:
                print('no csv file found with this date, checking next one')
        #if the json file has been created, not running for anymore dates
        else:
            break
        
def gen_subevent_bools(p_10,p_100):
    """
    given lists of peak fluxes for protons >10 MeV and >100 MeV, creates a boolean
    for whether or not each event is a subevent (doesn't cross a threshold)
    """
    subevent_bools = [None]*len(p_10)
    
    for j in range(len(p_10)):
        try:
            p10 = float(p_10[j])
        except ValueError:
            p10 = 'nan'
        
        try:
            p100 = float(p_100[j])
        except ValueError:
            p100 = 'nan'
        
        if str(p10) != 'nan' and str(p100) != 'nan':
            if p10 < 10 and p100 < 1:
                subevent_bools.append(True)
            else:
                if str(p10) == 'nan' and str(p100) != 'nan':
                    if p100 < 1:
                        subevent_bools.append(True)
                    else:
                        subevent_bools.append(False)
                elif str(p10) != 'nan' and str(p100) == 'nan':
                    if p10 < 10:
                        subevent_bools.append(True)
                    else:
                        subevent_bools.append(False)
                else:
                    subevent_bools.append(True)
        else:
            subevent_bools.append(True)
            
        return(subevent_bools)
        
def multi_events(start_time,end_time,subevent_bools):
    """
    takes in lists of start times, end times, and whether or not an event is a subevent,
    and uses those lists to run functions that subsequently extract data from the GOES
    database
    """
    
    for j in range(len(start_time)):

        st = start_time[j]
        et = end_time[j]
        subevent = subevent_bools[j]
        
        if str(st) != 'nan':
            try:
                st = parse(st)
                yes_st = True
            except ValueError:
                yes_st = False
        else:
            yes_st = False
        
        if str(et) != 'nan':
            try:
                et = parse(et)
                yes_et = True
            except ValueError:
                yes_et = False
        else:
            yes_et = False
    
        if yes_st and yes_et:
            if st > datetime(2010,9,1):
                try:
                    print('got start and end times! running Katies code')  
                    database_extraction(st,et,instrument_chosen,subevent)
                except:
                    continue
        