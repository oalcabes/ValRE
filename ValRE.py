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
for qualitative verification. To run ValRE, you must edit the configuration file
(config.py) included in the ValRE package.
"""

#%%
### LOADING LIBRARIES ###

import json as js
import os
from pathlib import Path
import numpy as np
from datetime import datetime,date,timedelta
from dateutil.parser import parse,isoparse
import config as cfg
from matplotlib import pyplot as plt, dates as mdates
from reportlab.lib import utils, colors
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet
from pandas.plotting import register_matplotlib_converters
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc


#%%
### SOME LIBRARY STUFF ###
register_matplotlib_converters()
styles = getSampleStyleSheet()

#%%
### DEFINING UNIVERSAL VARIABLES ###

#lists
num_thresholds = len(cfg.energy_threshold)
remove_threshold = []
rel_change = []
peak_diff_time = []
warning_times = []
observation_instruments = []

#reports
json_report = {}
report_elements=[]

#model info
model_name = cfg.model_name
mod_type = 'undef'

#contingency table values
hits = np.zeros(num_thresholds)
false_alarms = np.zeros(num_thresholds)
misses = np.zeros(num_thresholds)
correct_negatives = np.zeros(num_thresholds)

#values for ROC (Relative Operating Characteristic) graph
ROC_thresholds = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
ROC_h = np.zeros(len(ROC_thresholds))
ROC_m = np.zeros(len(ROC_thresholds))
ROC_fa = np.zeros(len(ROC_thresholds))
ROC_cn = np.zeros(len(ROC_thresholds))
ROC_H = np.zeros(len(ROC_thresholds))
ROC_F = np.zeros(len(ROC_thresholds))

#%%
### REPORT HEADERS ###

#creating header of PDF report
if cfg.PDF_report:
    model_name_report = Paragraph('%s Validation Report' %model_name,
                                  style = styles['title'])
    date_of_report = Paragraph('Date of report: ' + datetime.today().replace(microsecond=0).isoformat(),
                               style = styles['Normal'])
    validation_window_report = Paragraph('Validation window: ' +
                                         date(cfg.start_year,
                                              cfg.start_month,
                                              cfg.start_day).isoformat()
                                         + ' to ' + 
                                         str(date(cfg.end_year,
                                                  cfg.end_month,
                                                  cfg.end_day)),
                                         style = styles['Normal'])
    report_elements.append(model_name_report)
    report_elements.append(Spacer(1, 0.5*cm))
    report_elements.append(date_of_report)
    report_elements.append(validation_window_report)

#creating header of JSON report
if cfg.JSON_report:
    json_report['model_name'] = model_name
    json_report['date_of_report'] = datetime.today().replace(microsecond=0).isoformat()
    json_report['validation_window'] = {'start' : date(cfg.start_year,
                                                           cfg.start_month,
                                                           cfg.start_day).isoformat(),
                                        'end' : date(cfg.end_year,
                                                         cfg.end_month,
                                                         cfg.end_day).isoformat()}

    json_report['thresholds'] = [None]*num_thresholds

### CREATING OUTPUT DIRECTORIES IF NOT ALREADY PRESENT ###
    
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
    #header of reference sheet
    ref_out = Path(output_path)
    reference_elements=[]
    title = Paragraph('Validation Reference Sheet', style=styles['title'])
    reference_elements.append(title)

    #creating table with properties of metric scores
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
                    '0 to 1','1'],
                   ['Mean Percentage Error','','-infinity to infinity','0'],
                   ['Mean Absolute Percentage Error','','0 to infinity','0']) #check this value

    metric_ref_table=Table(metric_ref)
    metric_ref_table.setStyle(TableStyle([('BACKGROUND',(0,2),(3,13),
                                           colors.lightgrey),
                                          ('BACKGROUND',(0,15),(3,20),
                                           colors.lightgrey),
                                          ('GRID',(0,1),(3,20),0.25,
                                           colors.black),
                                          ('SPAN',(0,0),(3,0)),
                                          ('SPAN',(0,14),(3,14)),
                                          ('SPAN',(0,15),(1,15)),
                                          ('SPAN',(0,16),(1,16)),
                                          ('SPAN',(0,17),(1,17)),
                                          ('SPAN',(0,18),(1,18)),
                                          ('SPAN',(0,19),(1,19)),
                                          ('SPAN',(0,20),(1,20))]))

    #writing report
    reference_elements.append(metric_ref_table)

    reference = SimpleDocTemplate(str(ref_out / 'validation_reference.pdf'),
                                  pagesize=letter)
    reference.build(reference_elements)

def date_range(all_files,start_year,start_month,start_day,end_year,end_month,
               end_day):
    """
    from a list of files, extract only files within a given date range.
    inputs are integers.
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
    
    
def calc_all_clear_flux(peak_intensity,flux_threshold):
    """
    calculate an all clear value based on whether or not the peak flux has
    crossed the flux threshold
    """
    
    print('calculating all clear boolean using peak flux and threshold')
    if peak_intensity >= flux_threshold:
        all_clear_boolean = 'false'
    else:
        all_clear_boolean = 'true'
    
    print('all clear boolean calculated')
    
    return(all_clear_boolean)
                            
def calc_all_clear_prob(prob,prob_threshold):
    """
    calculate an all clear value based on whether or not the probability that an
    event will happen has crossed the probability threshold
    """
    
    print('calculating all clear boolean using probability and threshold')
    if prob >= prob_threshold:
        all_clear_boolean = 'false'
    else:
        all_clear_boolean = 'true'
    
    print('all clear boolean calculated')
    
    return(all_clear_boolean)
    
def determine_mod_type(all_forecasts,i):
    """
    determine whether a model is deterministic (has flux values), probabilistic,
    or neither based on whether or not flux or probability values are present in the
    model output
    """
    
    print('determining type of model')
    
    try:
        probability=(all_forecasts[i]['probabilities'][0]['probability_value'])
        mod_type = 'prob'
        print('model is probabilistic')
        
        return(mod_type)
        
    except KeyError:
        print('model is not probabilistic')
                        
    try:
        peak_intensity_model = (all_forecasts[i]['peak_intensity']['intensity'])
        mod_type = 'flux'
        print('model is deterministic')
        
        return(mod_type)
        
    except:
        print('model is not deterministic')
        
    mod_type = 'none'
    print('model is not probabilistic or deterministic')
    
    return(mod_type)
      
def extract_mod_values(all_forecasts,mod_type,i,flux_threshold,prob_threshold):
    """
    extract values needed for validation from a model file for a given proton energy
    threshold. if the model is probabilistic, probability, probability threshold, and all
    clear values will be extracted. if a model is deterministic, peak flux, peak time,
    and all clear values will be extracted. if a model is neither, only all clear values
    will be extracted.
    """
    
    mod_values = {"probability" : 'undef', "peak_flux" : "undef",
                  "peak_time" : "undef", "all_clear" : "undef", "type" : mod_type}
    
    #if model type is undefined, try to define it
    if mod_type ==  'undef':
        mod_values["type"] = determine_mod_type(all_forecasts,i)
        mod_type = mod_values["type"]
        
    #if model is probabilistic, extract probability values
    if mod_type == 'prob':
        mod_values["probability"] = float(all_forecasts[i]['probabilities'][0]
                                          ['probability_value'])

        print('model predicted probability values found')
        print('probability = %s' %mod_values["probability"])
        print('probability threshold = %s'
              %mod_values["probability_threshold"])

        print('extracting all clear boolean from model')
        
        #if model output has an all clear, extract it. if not, use probability
        #and threshold values to calculate it
        try:
            mod_values["all_clear"] = str(all_forecasts[i]['all_clear']
                                         ['all_clear_boolean']).lower()
            print('all clear extracted')
        except NameError:
            print('all clear not present in file - calculating')
            mod_values["all_clear"] = calc_all_clear_prob(probability,prob_threshold)
                                    
        print('all clear boolean model = %s' %mod_values["all_clear"])
        return(mod_values)
    #if model type is deterministic, extract flux values
    elif mod_type == 'flux':      
        mod_values["peak_flux"] = float(all_forecasts[i]['peak_intensity']['intensity'])                        
        print('model flux values found')
        print('model peak intensity = %s' %mod_values["peak_flux"])
                        
        #extracting peak times if they exist
        print('extracting peak time')
        try:
            mod_values["peak_time"] = (all_forecasts[mod_i]['peak_intensity']['time'])
            mod_values["peak_time"] = parse(mod_values["peak_time"])
                                    
            print('peak time found')
            print('peak time = %s' %mod_values["peak_time"])
        except:
            print('no peak time found') #keeping it as undefined

        #extracting all clear boolean if model has, if not calculating it
        print('extracting model all clear boolean')
        
        try:
            mod_values["all_clear"] = str(all_forecasts[mod_i]['all_clear']
                                         ['all_clear_boolean']).lower()
            print('all clear extracted')
        except KeyError:
            print('all clear not found - calculating using fluxes and threshold')
            mod_values["all_clear"] = calc_all_clear_flux(mod_values["peak_flux"],flux_threshold)

        print('all clear calculated')
        print('all clear boolean model = %s' %mod_values["all_clear"])
        
        return(mod_values)
    #if model is not deterministic or probabilistic, extracting all clear value
    elif mod_type == 'none':     
        try:
            mod_values["all_clear"] = str(all_forecasts[mod_i]['all_clear']['all_clear_boolean']).lower()
            print('all clear extracted')
            
            return(mod_values)         
        except:
            print('NO VALUES FOUND FOR THIS MODEL FILE')

#%%
### LOADING IN FILES ###
            
#getting path names
model_path = Path(cfg.model_path)
obs_path = Path(cfg.obs_path)

print('Loading in model and observation files')
#extracting all JSON files in given folders
model_files = [f for f in model_path.rglob("*.json")] #these are just file names
obs_files = [f for f in obs_path.rglob('*.json')]

print('Reducing to files from chosen date range')
#get only files within given date range
model_files = date_range(model_files,cfg.start_year,cfg.start_month,
                         cfg.start_day,cfg.end_year,cfg.end_month,cfg.end_day)
obs_files = date_range(obs_files,cfg.start_year,cfg.start_month,cfg.start_day,
                       cfg.end_year,cfg.end_month,cfg.end_day)

#%%
### COMPARING MODEL OUTPUT TO OBSERVATION OUTPUT (MAIN CODE) ###

#initializng plots
print('initializing plots')
peak_fig,peak_axes = plt.subplots(len(cfg.energy_threshold))
prob_fig,prob_axes = plt.subplots(len(cfg.energy_threshold))

print('Validation begun')

#comparing forecast and observed for each observed event
#assuming one observation file per event
for f in obs_files:
    print(f)
    if os.stat(f).st_size > 0: #choosing only files that aren't blank
        mod_event_files = [] #resetting list for corresponding model files
        
        #extracting observation output
        with open(f, 'r') as o:
            print('loading output')
            obs = js.load(o) #loading in json files
            obs_events = (obs['sep_forecast_submission']['triggers'][0]
                             ['particle_intensity']['ongoing_events'])
            instrument = (obs['sep_forecast_submission']['triggers'][0]
                             ['particle_intensity']['observatory'])
            
            #recording which instrument was used
            if instrument not in (observation_instruments):
                observation_instruments.append(instrument)
                
    else:
        print('observation file is blank - skipping')
            
    #looking at each energy channel present in observation file
    for i in range(len(obs_events)):
        energy_threshold = (obs_events[i]['energy_min'])
        print('event recorded for %s MeV energy channel' %energy_threshold)

        #only looking at forecasts for given energy channels
        if float(energy_threshold) in cfg.energy_threshold:
            print('energy channel wanted for validation - comparing '
                  'to model files')
            
            #figuring out what index in configuration lists  we're using
            j = cfg.energy_threshold.index(energy_threshold)
            
            #extracting given flux threshold based on observed energy level
            flux_threshold = cfg.pfu_threshold[j]
            
            #observed time values
            print('extracting observed start and end times')
            obs_start_time = (obs_events[i]['start_time'])
            obs_end_time = (obs_events[i]['end_time'])
            obs_start_time = parse(obs_start_time)
            obs_end_time = parse(obs_end_time)
            print('times extracted')
           
            #observed peak intensity values
            print('extracting observed peak intensities and peak times')
            peak_intensity_obs = float(obs_events[i]['peak_intensity'])
            peak_time_obs = (obs_events[i]['peak_time'])
            peak_time_obs = parse(peak_time_obs)
            print('peak values extracted')
           
            #getting observed all clear value
            print('extracting observed all clear')
            try:
                all_clear_boolean_obs = (obs_events[i]
                ['all_clear_boolean']) #all clear value
                print('all clear value extracted')
                print('all clear boolean observation = %s'
                      %all_clear_boolean_obs)
            except:
                print('all clear boolean not given - calculating')
                all_clear_boolean_obs = calc_all_clear_flux(peak_intensity_obs,flux_threshold)                  
            
            
            ### MODEL ###
            #finding all model files within given range before event starts
            print('finding model files that correspond to this observation')
            day_list=[]
            for d in range(cfg.days_before_event):
                day_list.append((obs_start_time - timedelta(days=d)).date())
            
            #creating list of these model files
            for day in day_list:
                for m in model_files:
                    if str(day) in str(m) or day.strftime('%Y_%m_%d') in str(m):
                        mod_event_files.append(m)
                       
            print('model files corresponding to event: %s' %mod_event_files)
            
            #checking to see if there actually are model files
            if len(mod_event_files) == 0:
                print('no model files found for this event')
                mod_record = False
            else:
                mod_record = True
            
            #creating lists for model values from all model files 
            mod_peak_list = []
            mod_peak_times = []
            mod_probs = []
            mod_prob_times = []
            all_clear_boolean_model = []
            mod_issue_times = []
            
            for mod_f in mod_event_files:
                #print(mod_f)
                with open(mod_f, 'r') as m:                           
                    mod_output = js.load(m)                        
                    all_forecasts = mod_output['sep_forecast_submission']['forecasts']
                    mod_issue_time = parse(mod_output['sep_forecast_submission']['issue_time'])                       
                
                #check to see if model has a forecast for the current energy threshold
                for k in range(len(all_forecasts)):
                    if all_forecasts[k]['energy_channel']['min'] == energy_threshold:
                        mod_i = k
                        mod_record = True
                        print('model has forecast for energy channel %s MeV' %energy_threshold)
                    else:
                        print('no model forecast for energy channel %s MeV' %energy_threshold)
                        mod_record = False
                
                #if model has a forecast for the energy channel, extract forecast
                if mod_record:
                    
                    #forecast start and end times
                    print('extracting forecast start and end times')
                    mod_start_time=(all_forecasts[mod_i]['prediction_window']
                                 ['start_time'])
                    mod_end_time=(all_forecasts[mod_i]['prediction_window']
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
                        
                    print('checking if model recorded event')
                    #checking to see if this file actually contains the event                   
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
        
                        #extract model values
                        mod_values = extract_mod_values(all_forecasts,mod_type,
                                                        mod_i,flux_threshold,
                                                        cfg.prob_threshold)                       
                        #determining what type of model it is
                        mod_type = mod_values["type"]                  
                        #adding model all clear value to all clear list
                        all_clear_boolean_model.append(mod_values["all_clear"])
                        mod_issue_times.append(mod_issue_time)
                     
                        #appending other values from the model to lists
                        if mod_type == 'flux':
                            mod_peak_list.append(float(mod_values["peak_flux"]))
                            #if no peak time is available, just making it the start time of the file
                            if mod_values["peak_time"] == "undef":
                                mod_peak_times.append(mod_start_time)
                            else:
                                mod_peak_times.append(mod_values["peak_time"])
                        elif mod_type == 'prob':
                            mod_probs.append(mod_values['prob'])
                            mod_prob_times.append(mod_start_time)
                else:
                    print('skipping file')

            #checking if any of the model files actually had viable all clear forecasts
            if len(all_clear_boolean_model) == 0:
                print('no model forecasts available for this event')
                mod_record = False
            else:
                mod_record = True
                print('list of model all clears: %s' %all_clear_boolean_model)
                             
#%% 
            #creating values for plots
            if mod_record:
                
                #forecast peak intensity value
                if mod_type == 'flux':
                    #if a threshold is crossed, extract the first flux value
                    #that crosses the threshold. otherwise extract the highest
                    if 'false' in all_clear_boolean_model:
                        peak_intensity_model = mod_peak_list[all_clear_boolean_model.index('false')]
                        peak_time_model = mod_peak_times[all_clear_boolean_model.index('false')]
                    else:
                        peak_intensity_model = max(mod_peak_list)
                        peak_time_model = mod_peak_times[mod_peak_list.index(peak_intensity_model)]                   
                #forecast probability value
                elif mod_type == 'prob':
                    #if a threshold is crossed, extract the first probability value
                    #that crosses the threshold. otherwise extract the highest
                    if 'false' in all_clear_boolean_model:
                        probability = mod_probs[all_clear_boolean_model.index('false')]
                        prob_time = mod_prob_times[all_clear_boolean_model.index('false')]
                    else:
                        probability = max(mod_probs)
                        prob_time = mod_prob_times[mod_probs.index(probability)]
                
                ### CREATING VALUE FOR CONTINGENCY TABLE ###
                print('creating value for contingency table')
                if str(all_clear_boolean_obs).lower() == 'false':
                    if 'false' in all_clear_boolean_model:
                        hits[j] = hits[j] + 1
                        notification_time = mod_issue_times[all_clear_boolean_model.index('false')]
                        warning_times.append(obs_start_time - notification_time)
                    else:
                        misses[j] = misses[j] + 1
                elif str(all_clear_boolean_obs).lower() == 'true':
                    if 'false' in all_clear_boolean_model:
                        false_alarms[j] = false_alarms[j]+1
                    else:
                        correct_negatives[j] = correct_negatives[j]+1
                
                #calculating relative change (later to be used when calculating mean
                #percent error and mean absolute percent error)
                try:
                    print('calculating peak intensity differences')
                    rel_change.append((float(peak_intensity_obs) - float(peak_intensity_model))/float(peak_intensity_obs))
                except:
                    print('incorrect data for peak difference calculations')
 
#%%
				### PLOTS ###

                print('generating plots')
                
                if mod_type == 'prob':
                    
                    #ROC plot
                    ROC_exist = True
                    print('generating values for ROC plot')
                    for thresh in ROC_thresholds:
                        t = ROC_thresholds.index(thresh)
                        #creating hits, false alarms, correct negatives, and misses
                        #for each ROC threshold
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
                                
                    #probability values plot
                    prob_graph = True
                    print('generating probability plot')
                    prob_axes[j].xaxis_date()
                    prob_axes[j].plot_date(prob_time,
                             float(probability),xdate=True,
                             ydate=False,marker='o',c='red',label='forecast')
                    prob_axes[j].text(1.05,0.5,'>' + str(cfg.energy_threshold[j])
                    + ' MeV\n' + str(cfg.pfu_threshold[j]) +
                                         ' pfu threshold',transform=prob_axes[j].transAxes)
                    if all_clear_boolean_obs == 'true':
                        obs_prob = 0
                    else:
                        obs_prob = 1
                    
                    prob_axes[j].plot_date(obs_start_time,float(obs_prob),xdate=True,
                                           ydate=False,marker='o',c='blue',
                                           label='observation')              
                    peak_graph = False
 
                elif mod_type == 'flux':
                    ROC_exist = False
                    prob_graph = False
                    peak_graph = True
                    
                    #peak values plot                
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
                    peak_axes[j].plot_date(peak_time_obs.date(),
                                           float(peak_intensity_obs),xdate=True,
                                           ydate=False,marker='o',c='blue',
                                           label='observation')
            else:
                if cfg.no_mod_file == 'all_clear':
                    all_clear_boolean_model = 'true'
                    print('creating value for contingency table')
                    if str(all_clear_boolean_obs).lower() == 'false':
                        misses[j] = misses[j] + 1
                    elif str(all_clear_boolean_obs).lower() == 'true':
                        correct_negatives[j] = correct_negatives[j]+1    
                else:
                    print('no model record found - skipping')
        else:
            print('energy channel not chosen for validation - skipping')
    
print('comparison between model and observation finished')

#%%
### METRICS CALCULATIONS ###

print('calculating metrics')

#average warning time
print('calculating average warning time')
avg_warning_time = np.mean(warning_times)
avg_warning_time = avg_warning_time - timedelta(microseconds=avg_warning_time.microseconds)
#adding average warning time to reports
if cfg.PDF_report:
    warning_time_report = Paragraph('Average Warning Time = ' +
                                    str(avg_warning_time), style = styles['Normal'])
    report_elements.append(warning_time_report)
    
if cfg.JSON_report:
    json_report['avg_warning_time'] = str(avg_warning_time)

#calculating mean percent error and mean absolute error if forecast fluxes available
if mod_type == 'flux':
    print('calculating mean percent error and mean absolute error')
    rel_change = np.array(rel_change)
    
    #mean percent error
    MPE = round((np.average(rel_change)),2)
    
    #mean absolute percent error
    MAPE = round((np.average(abs(rel_change))),2)

#calculating metric scores for each given threshold
for j in range(num_thresholds):
    print('calculating all other metrics for energy threshold %s MeV' %cfg.energy_threshold[j])

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

    #checking if model actually had data for this threshold
    if t == 0:
        print('no data for this threshold, no metric scores calculated' )
        threshold_avail = False
    else:
        threshold_avail = True

    #percent correct
    PC = round(((h + cn)/t),2)
    
    #bias
    bias = round(((h + fa)/(h+m)),2)

    #hit rate
    H = round((h/(h+m)),2)

    #false alarm rate
    F = round((fa/(fa+cn)),2)

    #frequency of misses
    FOM = round((m/(h+m)),2)

    #probability of correct negatives
    POCN = round((cn/(fa+cn)),2)

    #false alarm ratio
    FAR = round((fa/(h+fa)),2)

    #detection failure ratio
    DFR = round((m/(m+cn)),2)

    #frequency of correct negatives
    FOCN = round((cn/(m+cn)),2)

    #threat score
    TS = round((h/(h+fa+m)),2)

    #odds ratio
    OR = round(((h*cn)/(fa*m)),2)

    #reference scores
    hrand = ((h+fa)/t)*((h+m)/t)
    cnrand = ((m+cn)/t)*((fa+cn)/t)
    hpers = ((h+m)/t)**2
    cnpers = ((fa+cn)/t)**2

    #gilbert skill score
    M = TS
    Mref1 = Mref2 = hrand*(t/(h+fa+m))
    Mperf = 1
    GSS = round(((M - Mref1)/(Mperf - Mref2)),2)

    #true skill score
    M = PC
    Mref1 = hrand + cnrand
    Mref2 = hpers + cnpers
    Mperf = 1
    TSS = round((H - F),2)
    
    #Heidke skill score
    M = PC
    Mref1 = Mref2 = hrand + cnrand
    Mperf = 1
    HSS = round(((M - Mref1)/(Mperf - Mref2)),2)

    #odds ratio skill score
    ORSS = round((((h*cn)-(m*fa))/((h*cn)+(m*fa))),2)

#%%
### WRITING REPORTS ###
    
    if threshold_avail:
        print('writing validation report for %s MeV energy channel' %cfg.energy_threshold[j])

        #writing metrics into PDF report
        if cfg.PDF_report:
            
            if mod_type == 'prob' or mod_type == 'undef':
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
                                 ['Odds Ratio Skill Score', ORSS])
            elif mod_type == 'flux':
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
                                 ['Mean Percentage Error',MPE],
                                 ['Mean Absolute Percentage Error',MAPE])              

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
                                                 
            values = (['Contingency Table',''],['','Obs Yes','Obs No'],
                      ['Forecast Yes',h,fa],['Forecast No',m,cn])
            print(values)

            print('Contingency Table')

            t = Table(values)
            t.hAlign = 'LEFT'
            t.setStyle(TableStyle([('GRID',(0,1),(2,3),0.25,colors.black),
                                   ('BACKGROUND',(1,2),(1,2),colors.palegreen),
                                   ('BACKGROUND',(1,3),(1,3),colors.mistyrose),
                                   ('BACKGROUND',(2,2),(2,2),colors.mistyrose),
                                   ('BACKGROUND',(2,3),(2,3),colors.palegreen),
                                   ('SPAN',(0,0),(1,0))]))
           
            #putting the threshold into the report
            threshold_report = Paragraph('Energy Threshold = ' +
                                         str(cfg.energy_threshold[j]) +
                                         ' MeV<br />\nFlux Threshold = ' +
                                         str(cfg.pfu_threshold[j]) + ' pfu',
                                         style = styles['Normal'])

            #putting the instruments used for observations into the report
            instrument_report = Paragraph('Instruments used for observations: %s'
                                          %observation_instruments, style = styles['Normal'])
            
            #adding all of these objects to the report list
            report_elements.append(instrument_report)
            report_elements.append(threshold_report)
            report_elements.append(Spacer(1, 0.5*cm))
            report_elements.append(t)
            report_elements.append(Spacer(1, 1*cm))
            report_elements.append(metrics_report)
            report_elements.append(PageBreak())
            
        #writing metrics into json report
        if cfg.JSON_report:
            if mod_type == 'prob' or mod_type == 'undef':
                metrics_dict = {'Percent Correct' : PC, 'Bias' : bias, 'Hit Rate' : H,
                                'False Alarm Rate' : F, 'Frequency of Misses' : FOM,
                                'Probability of Correct Negatives' : POCN,
                                'False Alarm Ratio' : FAR,
                                'Detection Failure Ratio' : DFR,
                                'Frequency of Correct Negatives' : FOCN, 'Threat Score' : TS,
                                'Odds Ratio' : OR, 'True Skill Score' : TSS,
                                'Heidke Skill Score' : HSS, 'Odds Ratio Skill Score' : ORSS}
            elif mod_type == 'flux':
                metrics_dict = {'Percent Correct' : PC, 'Bias' : bias, 'Hit Rate' : H,
                                'False Alarm Rate' : F, 'Frequency of Misses' : FOM,
                                'Probability of Correct Negatives' : POCN,
                                'False Alarm Ratio' : FAR,
                                'Detection Failure Ratio' : DFR,
                                'Frequency of Correct Negatives' : FOCN, 'Threat Score' : TS,
                                'Odds Ratio' : OR, 'True Skill Score' : TSS,
                                'Heidke Skill Score' : HSS, 'Odds Ratio Skill Score' : ORSS,
                                'Mean Percentage Error' : MPE,
                                'Mean Absolute Percentage Error' : MAPE}
                
            json_report['obs_instruments'] = observation_instruments
            
            #threshold info
            json_report['thresholds'][j] = {}
            json_report['thresholds'][j]['energy_threshold'] = cfg.energy_threshold[j]
            json_report['thresholds'][j]['energy_unit'] = 'MeV'
            json_report['thresholds'][j]['flux_threshold'] = cfg.pfu_threshold[j]
            json_report['thresholds'][j]['flux_unit'] = 'pfu'

            #contingency table info
            json_report['thresholds'][j]['contingency_table'] = {}
            json_report['thresholds'][j]['contingency_table']['hits'] = h
            json_report['thresholds'][j]['contingency_table']['misses'] = m
            json_report['thresholds'][j]['contingency_table']['false_alarms'] = fa
            json_report['thresholds'][j]['contingency_table']['correct_negatives'] = cn

            #metric skills
            json_report['thresholds'][j]['metric_scores'] = metrics_dict
    #deleting subplots on graphs if model didn't have forecasts for this threshold
    else:
        print('deleting unused axes of graphs')
        if mod_type == 'prob':
            prob_fig.delaxes(prob_axes[j])
            prob_axes = prob_fig.axes
        elif mod_type == 'flux':
            peak_fig.delaxes(peak_axes[j])
            peak_axes = peak_fig.axes


#%%
### FINISHING PLOTS ###

print('finalizing plots')
if peak_graph:
    print('finalizing peak intensity plot')
    
    #labeling axes
    for ax in peak_axes:
        ax.xaxis_date()
        ax.set_ylabel('peak flux (pfu)')
        ax.set_yscale('log')
            #note: change this based on data
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%y'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    #creating a line at the threshold
    for l in range(len(peak_axes)):
        thresh_line = peak_axes[l].axhline(cfg.pfu_threshold[l],c='black',
                                      label='flux threshold')
    
    #generating a legend
    handles, labels = peak_fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    peak_axes[len(peak_axes) - 1].legend(by_label.values(), by_label.keys(),
                                         loc='lower right',
                                         bbox_transform=peak_axes[len(peak_axes)
                                         - 1].transAxes,
                                         bbox_to_anchor = (1.05,-0.8),ncol=3)
    
    #titles and x-axis labels
    peak_axes[0].set_title('Peak Proton Intensities for Chosen Events')
    peak_axes[len(peak_axes) - 1].set_xlabel('event time')

    #saving figure
    peak_fig.tight_layout()
    #saving as png
    peak_fig.savefig((png_out / (model_name + '_peak_intensities.png')),dpi=1000)
    
    #saving to pdf report
    peak_intensities = get_image((png_out / (model_name + '_peak_intensities.png')),
                                 width=15*cm)
    peak_intensities.hAlign = 'RIGHT'

else:
    print('no peak intensity graph created')

### ROC PLOT ###
if ROC_exist:
    print('finalizing ROC curve')
    
    #calculating Hit Rate and False Alarm Rate for each ROC threshold
    for t in range(len(ROC_thresholds)):
        ROC_H[t] = ROC_h[t]/(ROC_h[t]+ROC_m[t])
        ROC_F[t] = ROC_fa[t]/(ROC_fa[t]+ROC_cn[t])

    #actually plotting ROC graph
    ROC_fig, ROC_ax = plt.subplots()
    ROC_ax.plot(ROC_F,ROC_H)
    ROC_auc = auc(ROC_F,ROC_H)
    
    #generate RSS score
    RSS = 2*(ROC_auc - 0.5)
    print('ROC_F = %s' %ROC_F)
    print('ROC_H = %s' %ROC_H)
    print('RSS = %s' %RSS)
    
    if cfg.JSON_report:
        json_report['RSS'] = RSS
        
    ROC_ax.text(0.75, 0.75, 'RSS = %s' %RSS)
    ROC_ax.set_ylabel('Hit Rate')
    ROC_ax.set_xlabel('False Alarm Rate')
    ROC_ax.set_title('ROC curve')
    
    #saving figure and adding to PDF report
    ROC_fig.savefig((png_out / (model_name + '_ROC_curve.png')),dpi=1000)
    ROC_report = get_image((png_out / (model_name + '_ROC_curve.png')),width=10*cm)
else:
    if cfg.JSON_report:
        json_report['RSS'] = 'N/A'
    print('No ROC curve created')

if prob_graph:
    print('finalizing probability plot')
    for ax in prob_axes:
        ax.xaxis_date()
        ax.set_ylabel('probability')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%y'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    #creating horizontal lines for probability thresholds
    for l in range(len(prob_axes)):
        thresh_line = prob_axes[l].axhline(cfg.prob_threshold[l],c='black',
                                           label='probability threshold')

    #legend            
    handles, labels = prob_fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    prob_axes[len(prob_axes) - 1].legend(by_label.values(), by_label.keys(),
                                         loc='lower right',
                                         bbox_transform=prob_axes[len(prob_axes)
                                                                 - 1].transAxes,
                                         bbox_to_anchor = (1.5,-0.6),ncol=3)

    prob_axes[0].set_title('Probabilities for Chosen Events')
    prob_axes[len(prob_axes) - 1].set_xlabel('event time')
    prob_fig.subplots_adjust()
    prob_fig.tight_layout()
    
    #saving figure and adding to PDF report
    prob_fig.savefig((png_out / (model_name + '_probabilities.png')),
                     dpi=1000,bbox_inches='tight')
    probability_graph = get_image((png_out / (model_name + '_probabilities.png')),
                                  width=15*cm)
    probability_graph.hAlign = 'RIGHT'

else:
    print('no probability plot created')

print('total hits = %s' %(hits+cfg.man_hits))
print('total correct negatives = %s' %(correct_negatives+cfg.man_correct_negatives))
print('total false alarms = %s' %(false_alarms + cfg.man_false_alarms))
print('total misses = %s' %(misses+cfg.man_misses))

#%%

print('finalizing reports')



if cfg.PDF_report:
    #adding ROC graph to PDF report if it was created
    if ROC_exist:
        report_elements.append(ROC_report)

    #adding peak instensity graph to PDF report if it was created
    if mod_type == 'flux':
        report_elements.append(peak_intensities)
    
    #adding probability graph to PDF report if it was created
    if prob_graph:
        report_elements.append(probability_graph)
    
    #building PDF report
    pdf_name = '%s_validation.pdf' %model_name
    report = SimpleDocTemplate(str(val_out / pdf_name), pagesize=letter)
    report.build(report_elements)
    print('PDF report written')

#building JSON report
if cfg.JSON_report:
    json_name = '%s_validation.json' %model_name
    with open(val_out / json_name, 'w') as jsr:
        js.dump(json_report, jsr, indent=1)
    print('JSON report written')


#%%
### GENERATING REFERENCE SHEET ###
#if there isn't already a reference sheet present
if not (val_out / 'validation_reference.pdf').is_file():
    generate_ref_sheet(val_out)
    print('reference sheet created')
