#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:25:07 2019

@author: oalcabes
"""


"""
ValRE validates spaceweather forecasting models of SEP events. Validation is done by
comparing model forecasts of events to historical observations and generating skill scores,
which are written into reports (in PDF and/or JSON format). ValRE also creates plots
for qualitative verification.
"""

#%%
### LOADING LIBRARIES ###

import json as js
import os
from pathlib2 import Path
import numpy as np
from datetime import datetime,date,timedelta
from dateutil.parser import parse,isoparse
import pandas as pd
import config as cfg
from output_to_json import obs_csv2json
import operational_sep_quantities as sep
from matplotlib import pyplot as plt, dates as mdates
from reportlab.lib import utils, colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet
from pandas.plotting import register_matplotlib_converters
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
from urllib import request
from importlib import reload


#%%
### SOME LIBRARY STUFF ###
register_matplotlib_converters()
styles = getSampleStyleSheet()


#%%
### DEFINING UNIVERSAL VARIABLES ###

#lists
num_thresholds = len(cfg.energy_threshold)
remove_threshold = []
report_elements=[]
peak_diff_flux = []
peak_diff_time = []
json_report = {}
model_name = cfg.model_name

instrument_chosen = False

hits = np.zeros(num_thresholds)
false_alarms = np.zeros(num_thresholds)
misses = np.zeros(num_thresholds)
correct_negatives = np.zeros(num_thresholds)

#paths
ref_path = Path('ref_files')

#%%
### REPORT HEADERS ###

if cfg.PDF_report:
    model_name_report = Paragraph('%s Validation Report' %model_name,
                                  style = styles['title'])
    date_of_report = Paragraph('Date of report: ' + str(datetime.today()),
                               style = styles['Normal'])
    validation_window_report = Paragraph('Validation window: ' +
                                         str(datetime(cfg.start_year,cfg.start_month,
                                                      cfg.start_day))
                                         + ' to ' + 
                                         str(datetime(cfg.end_year,cfg.end_month,
                                                      cfg.end_day)),
                                         style = styles['Normal'])
    report_elements.append(model_name_report)
    report_elements.append(Spacer(1, 0.5*cm))
    report_elements.append(date_of_report)
    report_elements.append(validation_window_report)

if cfg.JSON_report:
    json_report['model_name'] = model_name
    json_report['date_of_report'] = str(datetime.today())
    json_report['validation_window'] = {'start' : str(datetime(cfg.start_year,
                                                               cfg.start_month,
                                                               cfg.start_day)),
                                        'end' : str(datetime(cfg.end_year,cfg.end_month,
                                                             cfg.end_day))}

json_report['thresholds'] = [None]*num_thresholds

### CREATING OUTPUT DIRECTORY ###

Path('validation_reports').mkdir(exist_ok=True)
val_out = Path('validation_reports')
Path(val_out / 'figures').mkdir(exist_ok=True)
png_out = val_out / Path('figures')

#%%
### FUNCTIONS ###

def get_image(path, width=1*cm):
    """
    Extract an image to a given size (specified by a given width) without
    altering width and height ratios
    """
    
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))

def generate_ref_sheet(output_path):
    """
    generate a PDF reference sheet specifying properties of metric scores
    """
    
    ref_out = Path(output_path)
    reference_elements=[]
    title = Paragraph('Validation Reference Sheet', style=styles['title'])
    reference_elements.append(title)

    #creating metric score table
    metric_ref = (['Metrics Reference Table','','',''],
                  ['Name', 'Attribute','Range','Perfect Score'],
                  ['Percent Correct','Accuracy','0 to 1', '1'],
                  ['Bias','Bias','0 to infinity','1'],
                  ['Hit Rate','Discrimination','0 to 1','1'],
                  ['False Alarm Rate','Discrimination','0 to 1', '0'],
                  ['Frequency of Misses','Discrimination','0 to 1','0'],
                  ['Probability of Correct Negatives','Discrimination',
                   '0 to 1','1'],
                  ['Frequency of Hits','Reliability and Resolution',
                   '0 to 1','1'],
                  ['False Alarm Ratio','Reliability and Resolution',
                   '0 to 1','1'],
                  ['Detection Failure Ratio','Reliability and Resolution',
                   '0 to 1','0'],
                  ['Frequency of Correct Negatives',
                   'Reliability and Resolution','0 to 1','1'],
                   ['Threat Score','Accuracy','0 to 1','1'],
                   ['Odds Ratio','Accuracy','0 to infinity','infinity'],
                   ['Skill Scores','','',''],
                   ['True Skill Score','','-1 to 1','1'],
                   ['Heidke Skill Score','','-1 to 1','1'],
                   ['Odds Ratio Skill Score','','-1 to 1','1'],
                   ['Relative Operating Characteristic Skill Score (RSS)','',
                    '0 to 1','1']) #check this value

    metric_ref_table=Table(metric_ref)
    metric_ref_table.setStyle(TableStyle([('BACKGROUND',(0,2),(3,13),
                                           colors.lightgrey),
                                          ('BACKGROUND',(0,15),(3,19),
                                           colors.lightgrey),
                                          ('GRID',(0,1),(3,18),0.25,
                                           colors.black),
                                          ('SPAN',(0,0),(3,0)),
                                          ('SPAN',(0,14),(3,14)),
                                          ('SPAN',(0,15),(1,15)),
                                          ('SPAN',(0,16),(1,16)),
                                          ('SPAN',(0,17),(1,17)),
                                          ('SPAN',(0,18),(1,18))]))

    #writing report
    reference_elements.append(metric_ref_table)

    reference = SimpleDocTemplate(str(ref_out / 'validation_reference.pdf'),
                                  pagesize=letter)
    reference.build(reference_elements)

#choosing files from a specific date range
def date_range(all_files,start_year,start_month,start_day,end_year,end_month,
               end_day):
    """
    from a list of files, extract only files within a given date range.
    inputs are numbers.
    """

    d1 = date(start_year,start_month,start_day)
    d_last = date(end_year,end_month,end_day)
    day_range = (d_last - d1).days
    #print('day range: %s' %day_range)
    files = []
    for t in range(day_range):
        d2 = d1 + timedelta(t)
        d2_str1 = str(d2)
        d2_str2 = d2.strftime('%Y_%m_%d')
        # print(d2)
        for f in all_files:
            if d2_str1 in str(f) or d2_str2 in str(f):
                files.append(f)
    return(files)

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
    
def database_extraction(mod_start_time,mod_end_time,instrument_chosen):
     #EXTRACTING OBS DATA USING KATIE'S CODE IF IT DOESN'T
     #EXIST currently - i think it isn't working bc no SEP
     #event happened (may need to fix)

    #extending prediction window bc sometimes it doesnt
    #work creating time window to use in katie's code
    window_end_date = (mod_end_time.date() + timedelta(days=1))
    window_start_date = (mod_start_time.date() - timedelta(days=1))

    print('determining if an instrument has been chosen')

    if instrument_chosen: #need to check if this works but cant :(
        if inst_end < window_end_date: #this is mad but its fine
            instrument_chosen = False

    else:
        try:
            #if instrument is specified in cfg using that
            instrument = cfg.instrument
            inst_end = datetime.today()
            print('using %s as our instrument for observations' %instrument)
            instrument_chosen = True

        except:
            #choosing instrument using function
            instrument_stuff = choose_prime_inst(window_start_date,
                                                 window_end_date)
            instrument = instrument_stuff[0]
            inst_end = instrument_stuff[1]
            instrument_chosen = True

        try:
            print('extracting data from GOES website')
            sep.run_all(str(window_start_date), str(window_end_date),
                        str(instrument), 'integral', '', '', True,
                        cfg.detect_previous_event, '100,1')
            
            reload(sep)

            print('extracted - reformatting')

            try:
                new_obs_name = ('sep_values_' + str(instrument) + '_integral_' +
                                mod_start_time.date().strftime('%Y_%m_%d')
                                .replace('_0','_') + '.csv')
                            
                obs_name = (str(instrument) + '_' +
                            str(mod_start_time.date()) + '.json')

                katies_path = Path('output')
                
                obs_csv2json((katies_path / new_obs_name), obs_name,
                              (ref_path/'example_sepscoreboard_json_file_v20190228.json'))
                print('obs file created')
                obs_record = True
                
            except FileNotFoundError:
                try:
                    new_obs_name = ('sep_values_' + str(instrument) + '_integral_' +
                                    window_start_date.strftime('%Y_%m_%d')
                                    .replace('_0','_') + '.csv')
                    obs_name = (str(instrument) + '_' +
                                str(mod_start_time.date()) + '.json')
                    katies_path = Path('output')
                
                    obs_csv2json((katies_path / new_obs_name), obs_name,
                                 (ref_path/'example_sepscoreboard_json_file_v20190228.json'))
                
                    print('obs file created')
                    obs_record = True
                
                except FileNotFoundError:
                    obs_record = False
                    print('no SEP event recorded, observed all clear = true')
                    all_clear_boolean_obs = 'true'              
                    return([obs_record,all_clear_boolean_obs,instrument])
                
            with open(obs_path / obs_name,'r') as o:
                obs = js.load(o)
                ('output files found - extracting values')
                #extracting values from output files
                obs_events = (obs['sep_forecast_submission']['triggers']
                                 [0]['particle_intensity']['ongoing_events'])
                
                return([obs_record,obs_events,instrument])
        
        except SystemExit:
            all_clear_boolean_obs = 'true'
            print('no SEP event recorded, observed all clear = true')
            obs_record = False
            
            return([obs_record,all_clear_boolean_obs,instrument])
    

#%%
### LOADING IN FILES ###

print('Validation begun')
model_path = Path(cfg.model_path)
obs_path = Path(cfg.obs_path)

print('Loading in model and observation files')
model_files = [f for f in model_path.rglob("*.json")] #these are just file names
obs_files = [f for f in obs_path.rglob('*.json')]

print('Extracting files from chosen date range')
model_files = date_range(model_files,cfg.start_year,cfg.start_month,
                         cfg.start_day,cfg.end_year,cfg.end_month,cfg.end_day)

#%%
### COMPARING MODEL OUTPUT TO OBSERVATION OUTPUT ###

peak_fig,peak_axes = plt.subplots(len(cfg.energy_threshold))
prob_fig,prob_axes = plt.subplots(len(cfg.energy_threshold))

print('Comparing model output to observation output')
for f in model_files:
    print(f)
    if os.stat(f).st_size > 0: #choosing only files that aren't blank
        print('checked blank')
        with open(f, 'r') as d:
            print('loading output')
            output = js.load(d) #loading in json files
            all_forecasts = output['sep_forecast_submission']['forecasts']

            #looking at each forecast
            for i in range(len(all_forecasts)):
                print('going through thresholds')

                energy_threshold = (all_forecasts[i]['energy_channel']['min'])
                print('energy_threshold = %s' %energy_threshold)

                #extracting forecasts for given energy levels
                if energy_threshold in cfg.energy_threshold:

                    #given energy threshold index
                    j = cfg.energy_threshold.index(energy_threshold)

                    print('extracting model start and end times')
                    #extracting start and end time values - DEAL WITH THIS
                    mod_start_time=(all_forecasts[i]['prediction_window']
                                    ['start_time'])
                    mod_end_time=(all_forecasts[i]['prediction_window']
                                  ['end_time'])

                    #putting start and end times in datetime formats

                    try:
                        mod_start_time = parse(mod_start_time)
                        mod_end_time = parse(mod_end_time)
                        print('start and end times extracted')
                    except:
                        print('start and end times invalid - fixing format')
                        mod_start_time = str(mod_start_time).replace('Z','T',0)
                        print(mod_start_time)
                        mod_start_time = isoparse(mod_start_time)
                        mod_end_time = str(mod_end_time).replace('Z','T',0)
                        mod_end_time = isoparse(mod_end_time)
                        print('start and end time formats fixed, extracted')

                    try:
                        #extracting probability values (if the model has)
                        probability=(all_forecasts[i]['probabilities'][0]
                                    ['probability_value'])
                        probability_type_model = True

                    except KeyError:
                        probability_type_model = False
                        
                    try:
                        peak_intensity_model = (all_forecasts[i]
                                                ['peak_intensity']
                                                ['intensity'])
                        peak_type_model = True
                    except:
                        peak_type_model = False

                    if probability_type_model:
                        uncertainty=(all_forecasts[i]['probabilities'][0]
                                     ['uncertainty'])
                        probability_threshold=(all_forecasts[i]['all_clear']
                                              ['probability_threshold'])

                        #noting that the model has a probabilistic forecast


                        print('model predicted probability values found')
                        print('probability = %s' %probability)
                        print('uncertainty = %s' %uncertainty)
                        print('probability threshold = %s'
                              %probability_threshold)

                        print('extracting all clear boolean from model')
                        try:
                            #if model output has an all clear, extract it
                            all_clear_boolean_model = (all_forecasts[i]
                                                       ['all_clear']
                                                       ['all_clear_boolean'])
                            print('all clear extracted')

                        except NameError:
                            print('all clear not present in file - '
                                  'calculating')
                            if float(probability)>float(probability_threshold):
                                all_clear_boolean_model = 'false'
                            else:
                                all_clear_boolean_model = 'true'
                            print('all clear calculated')

                        print('all clear boolean model = %s'
                              %all_clear_boolean_model)

                    elif peak_type_model:
                        #extracting peak intensity values if the model
                        #doesn't have probability values
                        
                        print('model flux values found')
                        print('model peak intensity = %s'
                              %peak_intensity_model)
                        
                            #extracting peak times if they exist- if not just
                            #using start time
                        print('extracting peak time')

                        #NOTE: PROBABLY CHANGE THIS WHEN I FIX START TIME STUFF
                        try:
                            peak_time_model = (all_forecasts[i]
                                               ['peak_intensity']
                                               ['time'])
                            peak_time_model = parse(peak_time_model)

                            print('peak time found')
                            print('peak time = %s' %peak_time_model)

                        except:
                            print('no peak time found - using start time'
                                  ' instead')
                            peak_time_model = (mod_start_time)
                            
                        print('peak time model = %s' %peak_time_model)

                        #extracting all clear boolean if model has, if not
                        #calculating it
                        print('extracting model all clear boolean')
                        try:
                            all_clear_boolean_model = (all_forecasts[i]
                                                       ['all_clear']
                                                       ['all_clear_boolean'])
                            print('all clear extracted')

                        except:
                        #calculating all clear boolean using peak intensities
                        #and thresholds
                        #(it may be possible to do this more efficiently)
                            print('all clear not found - calculating using'
                                  'fluxes and threshold')
                            flux_threshold = (cfg.pfu_threshold
                                              [cfg.energy_threshold.index
                                               (energy_threshold)])

                            if peak_intensity_model >= flux_threshold:
                                all_clear_boolean_model = 'false'
                            else:
                                all_clear_boolean_model = 'true'
                            print('all clear calculated')

                            print('all clear boolean model = %s'
                                  %all_clear_boolean_model)
                    else:
                        all_clear_boolean_model = (all_forecasts[i]['all_clear']
                                                   ['all_clear_boolean'])
                        print('all clear extracted')

                    #generating name of corresponding obs file
                    #will likely need to change this to make it more
                    #generalized - unless if i use my previous code to generate
                    #observation data then it's good

                    #extracting obs file from obs output folder
                    print('extracting observation files from observation'
                          'output folder')
                    #may need to change some stuff abt the obs name

                    if ((str(mod_start_time.date()) in str(obs_files)) or
                    (mod_start_time.strftime('%Y_%m_%d') in str(obs_files))):
                        for obs_f in obs_files:
                            if ((str(mod_start_time.date()) in str(obs_f)) or
                            (mod_start_time.strftime('%Y_%m_%d') in str(obs_f))):
                                with open(obs_f,'r') as o:
                                    obs = js.load(o)
                                    ('output files found - extracting values')
                                    #extracting values from output files
                                    obs_events = (obs['sep_forecast_submission']
                                                  ['triggers'][0]
                                                  ['particle_intensity']
                                                  ['ongoing_events'])
                                    instrument = (obs['sep_forecast_submission']
                                                 ['triggers'][0]
                                                 ['particle_intensity']
                                                 ['observatory'])
                                    obs_record = True
                                

                    else:
                        print('observation files not found -'
                              'extracting data from GOES database')
                        db = database_extraction(mod_start_time,mod_end_time,
                                                 instrument_chosen)
                        
                        obs_record = db[0]
                        instrument = db[2]
                    
                        if obs_record:
                            obs_events = db[1]
                            obs_files = [f for f in obs_path.rglob('*.json')]
                        else:
                            all_clear_boolean_obs = db[1]

                        print('extracting values from new data files')

                    if obs_record:
                        for k in range (len(obs_events)):
                            obs_energy_threshold = obs_events[k]['energy_min']
                            if obs_energy_threshold == energy_threshold:
                                break

                            #observed time values
                        print('extracting observed start and end times')
                        obs_start_time = (obs_events[k]['start_time'])
                        obs_end_time = (obs_events[k]['end_time'])

                        obs_start_time = parse(obs_start_time)
                        obs_end_time = parse(obs_end_time)
                        print('times extracted')


                        print('extracting observed peak intensities and peak times')

                        #observed peak intensity values
                        peak_intensity_obs = (obs_events[k]['peak_intensity'])
                        peak_time_obs = (obs_events[k]['peak_time'])
                        peak_time_obs = parse(peak_time_obs)

                        print('peak values extracted')
                        
                        if mod_end_time < obs_start_time:
                            all_clear_boolean_obs = 'true'
                            
                        else:
                            #observed all clear boolean
                            print('extracting observed all clear')
                            try:
                                all_clear_boolean_obs = (obs_events[k]
                                ['all_clear_boolean']) #all clear value
                                print('all clear value extracted')
                                print('all clear boolean observation = %s'
                                  %all_clear_boolean_obs)
                            except:
                                print('all clear boolean not given - calculating')
                                if peak_intensity_obs >= flux_threshold:
                                    all_clear_boolean_obs = 'false'
                                else:
                                    all_clear_boolean_obs = 'true'
                                print('all clear boolean calculated')
                                print('all clear boolean observation'
                                      '= %s' %all_clear_boolean_obs)

                        print('data extracted')

#%%
                ### CREATING VALUES FOR CONTINGENCY TABLES ###
                    print('creating values for contingency table')
                    if str(all_clear_boolean_obs).lower() == 'false':
                        if str(all_clear_boolean_model).lower() == 'false':
                            hits[j] = hits[j]+1
                        elif str(all_clear_boolean_model).lower() == 'true':
                            misses[j] = misses[j]+1
                    elif str(all_clear_boolean_obs).lower() == 'true':
                        if str(all_clear_boolean_model).lower() == 'false':
                            false_alarms[j] = false_alarms[j]+1
                        elif str(all_clear_boolean_model).lower() == 'true':
                            correct_negatives[j] = correct_negatives[j]+1

                    try:
                        peak_diff_flux.append(float(peak_intensity_model) -
                                             float(peak_intensity_obs))
                    except:
                        print('incorrect data for peak difference calculations')

#%%
				### PLOTS ###

                    print('generating comparison plots')

                ### GENERATING VALUES FOR ROC PLOT ###
                    if probability_type_model:
                        ROC_exist = True
                        print('generating ROC plot')
                        ROC_thresholds = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
                        ROC_h = np.zeros(len(ROC_thresholds))
                        ROC_m = np.zeros(len(ROC_thresholds))
                        ROC_fa = np.zeros(len(ROC_thresholds))
                        ROC_cn = np.zeros(len(ROC_thresholds))
                        ROC_H = np.zeros(len(ROC_thresholds))
                        ROC_F = np.zeros(len(ROC_thresholds))
                        for thresh in ROC_thresholds:
                            t = ROC_thresholds.index(thresh)
                            print(thresh)
                            print('observed all clear: %s' %all_clear_boolean_obs)
                            print('model probability: %s' %probability)
                            if str(all_clear_boolean_obs).lower() == 'false':
                                if float(probability) >= thresh:
                                    ROC_h[t] = ROC_h[t] + 1
                                    print(ROC_h[t])
                                else:
                                    ROC_m[t] = ROC_m[t] + 1
                                    print(ROC_m[t])
                            else:
                                if float(probability) >= thresh:
                                    ROC_fa[t] = ROC_fa[t] + 1
                                    print(ROC_fa[t])
                                else:
                                    ROC_cn[t] = ROC_cn[t] + 1
                                    print(ROC_cn[t])


                    ### PLOTTING PROBABILITY VALUES ###

                        prob_graph = True
                        print('generating probability plot')
                        prob_axes[j].xaxis_date()
                        prob_axes[j].plot_date(mod_start_time,
                                          float(probability),xdate=True,
                                          ydate=False,marker='o',c='red',label='forecast')
                        prob_axes[j].text(1.05,0.5,'>' + str(cfg.energy_threshold[j])
                                         + ' MeV\n' + str(cfg.pfu_threshold[j]) +
                                         ' pfu threshold',transform=prob_axes[j].transAxes)

                        if all_clear_boolean_obs == 'true':
                            obs_prob = 0
                        else:
                            obs_prob = 1

                        if obs_record:
                            prob_axes[j].plot_date(obs_start_time,float(obs_prob),
                                                  xdate=True,ydate=False,marker='o',
                                                  c='blue',label='observation')
                        else:
                            prob_axes[j].plot_date(mod_start_time,float(obs_prob),
                                                  xdate=True,ydate=False,marker='o',
                                                  c='blue',label='observation')

                        peak_graph = False

                    elif peak_type_model:

                        ### PLOTTING PEAK INTENSITY VALUES ###

                        ROC_exist = False
                        prob_graph = False
                        peak_graph = True

                        print('generating peak intensity plot')
                        peak_axes[j].xaxis_date()
                        peak_axes[j].plot_date(peak_time_model.date(),
                                              float(peak_intensity_model),xdate=True,
                                              ydate=False,marker='o',c='red',
                                              label='forecast')
                        peak_axes[j].text(1.05,0.5,'>' + str(cfg.energy_threshold[j])
                                         + ' MeV\n' + str(cfg.pfu_threshold[j]) +
                                         ' pfu threshold',
                                         transform=peak_axes[j].transAxes)

                        #this is last in case obs peak intensity doesnt exist
                        if obs_record:
                            peak_axes[j].plot_date(peak_time_obs.date(),
                                                   float(peak_intensity_obs),xdate=True,
                                                   ydate=False,marker='o',c='blue',
                                                   label='observation')

print('comparison between model and observation finished')



#%%
### METRICS CALCULATIONS ###

print('calculating metrics')

if peak_type_model:
    avg_diff_flux = np.average(peak_diff_flux)
    #avg_diff_time = sum(peak_diff_time, timedelta(0)) / len(peak_diff_time)
    #check that this calculation is correct
    #be careful with time differences bc i think my peak time stuff is not always perfect
            
    peak_diffs_report = Paragraph('Average Peak Flux Difference = ' +
                                  str(avg_diff_flux) + ' pfu', style = styles['Normal']) #<br />) +
                                  #'Average Peak Time Difference = '
                                  #+ str(avg_diff_time),
                                  #)
    report_elements.append(peak_diffs_report)

#need to put in which threshold im using here
for j in range(num_thresholds):

#hits
    h = hits[j] + cfg.man_hits[j]
#false alarms
    fa = false_alarms[j] + cfg.man_false_alarms[j]
#misses
    m = misses[j] + cfg.man_misses[j]
#correct negatives
    cn = correct_negatives[j] + cfg.man_correct_negatives[j]
#total
    t = h + fa + m + cn

    print(t)

    if t == 0:
        print('no data for this threshold' )
        threshold_avail = False
        #maybe say something on report about this? idk - or have a final notes section
        #that prints and says all of the things that failed
    else:
        threshold_avail = True

#percent correct
    PC = (h + cn)/t
    bias = (h + fa)/(h+m)

#hit rate
    H = h/(h+m)

#false alarm rate
    F = fa/(fa+cn)

#frequency of misses
    FOM = m/(h+m)

#probability of correct negatives
    POCN = cn/(fa+cn)

#false alarm ratio
    FAR = fa/(h+fa)

#detection failure ratio
    DFR = m/(m+cn)

#frequency of correct negatives
    FOCN = cn/(m+cn)

#threat score
    TS = h/(h+fa+m)

#odds ratio
    OR = (h*cn)/(fa*m)

#reference scores
    hrand = ((h+fa)/t)*((h+m)/t)
    cnrand = ((m+cn)/t)*((fa+cn)/t)
    hpers = ((h+m)/t)**2
    cnpers = ((fa+cn)/t)**2

    #gilbert skill score
    M = TS
    Mref1 = Mref2 = hrand*(t/(h+fa+m))
    Mperf = 1
    GSS = (M - Mref1)/(Mperf - Mref2)

#true skill score
#not sure if i need these M values
    M = PC
    Mref1 = hrand + cnrand
    Mref2 = hpers + cnpers
    Mperf = 1
    TSS = H - F
#test the longer answers to make sure you get the same stuff

#Heidke skill score
    M = PC
    Mref1 = Mref2 = hrand + cnrand
    Mperf = 1
    HSS = (M - Mref1)/(Mperf - Mref2)

#odds ratio skill score
    ORSS = ((h*cn)-(m*fa))/((h*cn)+(m*fa))

    RSS = 'N/A' #change this

    if threshold_avail:

        if cfg.PDF_report:

            metrics_array = (['METRIC SCORES',''],
                             ['Percent Correct',PC], ['Bias', bias],
                             ['Hit Rate', H], ['False Alarm Rate',F,],
                             ['Frequency of Misses', FOM],
                             ['Probability of Correct Negatives',POCN],
                             ['False Alarm Ratio', FAR], ['Detection Failure Ratio',DFR],
                             ['Frequency of Correct Negatives', FOCN],
                             ['Threat Score', TS],
                             ['Odds Ratio', OR], ['True Skill Score', TSS],
                             ['Heidke Skill Score',HSS],
                             ['Odds Ratio Skill Score', ORSS],
                             ['Relative Operating Characteristic Skill Score',RSS])

            metrics_report = Table(metrics_array)
            metrics_report.hAlign = 'LEFT'
            metrics_report.setStyle(TableStyle([('BACKGROUND', (0,0),
                                                 (1,len(metrics_array)),
                                                 colors.lightgrey),('INNERGRID', (0,0),
                                                 (1,len(metrics_array)),0.25,
                                                 colors.black),
                                                 ('BOX', (0,0), (1,len(metrics_array)),
                                                  0.25,colors.black),
                                                 ('SPAN',(0,0),(1,0))]))

        if cfg.JSON_report:
            metrics_dict = {'Percent Correct' : PC, 'Bias' : bias, 'Hit Rate' : H,
                            'False Alarm Rate' : F, 'Frequency of Misses' : FOM,
                            'Probability of Correct Negatives' : POCN,
                            'False Alarm Ratio' : FAR,
                            'Detection Failure Ratio' : DFR,
                            'Frequency of Correct Negatives' : FOCN, 'Threat Score' : TS,
                            'Odds Ratio' : OR, 'True Skill Score' : TSS,
                            'Heidke Skill Score' : HSS, 'Odds Ratio Skill Score' : ORSS,
                            'Relative Operating Characteristic Skill Score' : RSS}


#%%

        print('writing validation report')

        if cfg.PDF_report:
            values = (['Contingency Table',''],
                      ['','Obs Yes','Obs No'],
                      ['Forecast Yes',h,fa],
                      ['Forecast No',m,cn])
            print(values)

            print('Contingency Table')
        #df = pd.DataFrame(values, index=['mod_yes','mod_no'], columns =
        #['obs_yes','obs_no'])
        #rint(df)

            t = Table(values)
            t.hAlign = 'LEFT'
            t.setStyle(TableStyle([('GRID',(0,1),(2,3),0.25,colors.black),
                                   ('BACKGROUND',(1,2),(1,2),colors.palegreen),
                                   ('BACKGROUND',(1,3),(1,3),colors.mistyrose),
                                   ('BACKGROUND',(2,2),(2,2),colors.mistyrose),
                                   ('BACKGROUND',(2,3),(2,3),colors.palegreen),
                                   ('SPAN',(0,0),(1,0))]))

       #JSC_skill_score = Paragraph('skill score = 1000',style = styles['Normal'])
       #may want to create something like a JSC skill score idk
            threshold_report = Paragraph('Energy Threshold = ' +
                                         str(cfg.energy_threshold[j]) +
                                         ' MeV<br />\nFlux Threshold = ' +
                                         str(cfg.pfu_threshold[j]) + ' pfu',
                                         style = styles['Normal'])

            instrument_report = Paragraph('Instrument used for observations: %s'
                                          %instrument, style = styles['Normal'])
            
            #report_elements.append(JSC_skill_score)
            report_elements.append(instrument_report)
            report_elements.append(threshold_report)
            report_elements.append(Spacer(1, 0.5*cm))
            report_elements.append(t)
            report_elements.append(Spacer(1, 1*cm))
            report_elements.append(metrics_report)

            report_elements.append(PageBreak())

        if cfg.JSON_report:
            json_report['obs_instrument'] = instrument
            #should i print this multiple times?

            json_report['thresholds'][j] = {}
            json_report['thresholds'][j]['energy_threshold'] = cfg.energy_threshold[j]
            json_report['thresholds'][j]['energy_unit'] = 'MeV'
            json_report['thresholds'][j]['flux_threshold'] = cfg.pfu_threshold[j]
            json_report['thresholds'][j]['flux_unit'] = 'pfu'

            json_report['thresholds'][j]['contingency_table'] = {}
            json_report['thresholds'][j]['contingency_table']['hits'] = h
            json_report['thresholds'][j]['contingency_table']['misses'] = m
            json_report['thresholds'][j]['contingency_table']['false_alarms'] = fa
            json_report['thresholds'][j]['contingency_table']['correct_negatives'] = cn

        #probably check this
            json_report['thresholds'][j]['metric_scores'] = metrics_dict

    else:
        if probability_type_model:
            prob_fig.delaxes(prob_axes[j])
            prob_axes = prob_fig.axes
            #prob_fig.subplots_adjust(bottom=0.5)
            #prob_fig.frameon = False
        elif peak_type_model:
            peak_fig.delaxes(peak_axes[j])
            peak_axes = peak_fig.axes


#%%
### FINISHING PLOTS ###

print('finalizing plots')
if peak_graph:
    print('finalizing peak intensity plot')
    for ax in peak_axes:
        ax.xaxis_date()
        ax.set_ylabel('peak flux (pfu)')
        ax.set_yscale('log')
            #note: change this based on data
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%y'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    for l in range(len(cfg.pfu_threshold)):
        thresh_line = peak_axes[l].axhline(cfg.pfu_threshold[l],c='black',
                                      label='flux threshold')

    handles, labels = peak_fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    peak_axes[len(peak_axes) - 1].legend(by_label.values(), by_label.keys(),
                                         loc='lower right',
                                         bbox_transform=peak_axes[len(peak_axes)
                                         - 1].transAxes,
                                         bbox_to_anchor = (1.05,-0.8),ncol=3)
#note: legend is not fully in figure for some reason
    peak_axes[0].set_title('Peak Proton Intensities for Chosen Events')
    peak_axes[len(peak_axes) - 1].set_xlabel('event time')

    #legend2 = ax.legend()
    peak_fig.tight_layout()
    peak_fig.savefig((png_out / (model_name + '_peak_intensities.png')),dpi=1000)

    peak_intensities = get_image((png_out / (model_name + '_peak_intensities.png')),
                                 width=15*cm)
    peak_intensities.hAlign = 'RIGHT'

else:
    print('no peak intensity graph created')

### ROC PLOT ###
#need to test ROC stuff to make sure it actually works (put in fake numbers)
#does this stuff come later?? i think so
if ROC_exist:
    print('finalizing ROC curve')
    for t in range(len(ROC_thresholds)):
        ROC_H[t] = ROC_h[t]/(ROC_h[t]+ROC_m[t])
        ROC_F[t] = ROC_fa[t]/(ROC_fa[t]+ROC_cn[t])

    ROC_fig, ROC_ax = plt.subplots()
    ROC_ax.plot(ROC_F,ROC_H)
    ROC_auc = auc(ROC_F,ROC_H)
    RSS = 2*(ROC_auc - 0.5)
    print('ROC_F = %s' %ROC_F)
    print('ROC_H = %s' %ROC_H)
    print('RSS = %s' %RSS)
    ROC_ax.set_ylabel('Hit Rate')
    ROC_ax.set_xlabel('False Alarm Rate')
    ROC_ax.set_title('ROC curve')
    print('ROC_auc = %s' %ROC_auc)
    ROC_fig.savefig((png_out / (model_name + '_ROC_curve.png')),dpi=1000)
    ROC_report = get_image((png_out / (model_name + '_ROC_curve.png')),width=10*cm)
else:
    RSS = 'N/A'
    print('No ROC curve created')

if prob_graph:
    print('finalizing probability plot')
    for ax in prob_axes:
        ax.xaxis_date()
        ax.set_ylabel('probability')
            #note: change this based on data
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%y'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    for l in range(len(prob_axes)):
        try:
            thresh_line = prob_axes[l].axhline(cfg.prob_threshold[l],c='black',
                                      label='probability threshold')
        except:
            thresh_line = prob_axes[l].axhline(probability_threshold,c='black',
                                      label='probability threshold')
#a lot of this stuff is not appearing and i dont know why
    handles, labels = prob_fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    prob_axes[len(prob_axes) - 1].legend(by_label.values(), by_label.keys(),
                                         loc='lower right',
                                         bbox_transform=prob_axes[len(prob_axes)
                                                                 - 1].transAxes,
                                         bbox_to_anchor = (1.5,-0.6),ncol=3)
#note: legend is not fully in figure for some reason
    prob_axes[0].set_title('Probabilities for Chosen Events')
    prob_axes[len(prob_axes) - 1].set_xlabel('event time')
    prob_fig.subplots_adjust()
    prob_fig.tight_layout()
    prob_fig.savefig((png_out / (model_name + '_probabilities.png')),dpi=1000,bbox_inches='tight')

    probability_graph = get_image((png_out / (model_name + '_probabilities.png')),width=15*cm)
    probability_graph.hAlign = 'RIGHT'

else:
    print('no probability plot created')

print('total hits = %s' %(hits+cfg.man_hits))
print('total correct negatives = %s' %(correct_negatives+cfg.man_correct_negatives))
print('total false alarms = %s' %(false_alarms + cfg.man_false_alarms))
print('total misses = %s' %(misses+cfg.man_misses))

#%%

print('finalizing report')

if ROC_exist:
    report_elements.append(ROC_report)

if peak_type_model:
    report_elements.append(peak_intensities)

if prob_graph:
    report_elements.append(probability_graph)

if cfg.PDF_report:
    pdf_name = '%s_validation.pdf' %model_name
    report = SimpleDocTemplate(str(val_out / pdf_name), pagesize=letter)
    report.build(report_elements)
    print('PDF report written')

if cfg.JSON_report:
    json_name = '%s_validation.json' %model_name
    with open(val_out / json_name, 'w') as jsr:
        js.dump(json_report, jsr, indent=1)
    print('JSON report written')


#%%
### GENERATING REFERENCE SHEET ###

if not (val_out / 'validation_reference.pdf').is_file():
    generate_ref_sheet(val_out)
    print('reference sheet created')
