#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:19:02 2019

@author: oliviaalcabes
"""

#import numpy as np
import csv
import glob
import json as js
import pprint
from pathlib import Path
from datetime import time,date,datetime
import config as cfg

#NOW : TEST TO SEE IF GENERALIZED CODE ACTUALLY WORKED - IF SO DELETE ALL OLD USELESS STUFF

#converting sepmod csv files from katie's code to json function
def sepmod_csv2json(input_file,output_file,example_path):

    model_path = Path(cfg.model_path)

    fieldnames = ('energy_threshold','flux_threshold','start_time','intensity','peak_time','rise_time','end_time','duration','fluence>10','fluence>100')

    with open(input_file,'r') as f:
        reader = csv.DictReader(f, fieldnames)
        out = js.dumps( [ row for row in reader ] )
    #for row in reader:
    #    out = js.dumps(row)
        #print(out)



    sepmod_data = js.loads(out)
#pprint.pprint(sepmod_data)

    with open(example_path,'r') as e:
        example = js.load(e)

#deleting unused categories
    del(example['sep_forecast_submission']['triggers'])
    example['sep_forecast_submission']['model']['short_name'] = 'SEPMOD' #need to find some way to generalize this maybe?
    del(example['sep_forecast_submission']['forecasts'][2])
    del(example['sep_forecast_submission']['forecasts'][0]['probabilities'])
    del(example['sep_forecast_submission']['forecasts'][0]['peak_intensity']['uncertainty'])
    del(example['sep_forecast_submission']['forecasts'][0]['peak_intensity_esp'])
    del(example['sep_forecast_submission']['forecasts'][1]['probabilities'])
    del(example['sep_forecast_submission']['forecasts'][0]['all_clear']['probability_threshold'])
    del(example['sep_forecast_submission']['forecasts'][1]['all_clear']['probability_threshold'])

#adding in values
    peak_intensity = { 'intensity' : 'intensity_value', 'units' : 'pfu', 'time' : 'peak_time'}
    example['sep_forecast_submission']['forecasts'][1]['peak_intensity'] = peak_intensity
    fluence = []
    fluence.append({'energy_min' : '10', 'fluence_value' : 'fluence_value', 'units' : 'MeV [cm^-2]'})
    fluence.append({'energy_min' : '100', 'fluence_value' : 'fluence_value', 'units' : 'MeV [cm^-2]'})
    example['sep_forecast_submission']['forecasts'][0]['fluence'] = fluence
    example['sep_forecast_submission']['forecasts'][1]['fluence'] = fluence

#things that i am deleting now but will probably need to add in:
    del(example['sep_forecast_submission']['contacts'])
    del(example['sep_forecast_submission']['model']['spase_id'])
    del(example['sep_forecast_submission']['forecasts'][0]['event_length']['threshold'])
    del(example['sep_forecast_submission']['forecasts'][0]['event_length']['threshold_units'])
    del(example['sep_forecast_submission']['forecasts'][1]['event_length']['threshold'])
    del(example['sep_forecast_submission']['forecasts'][1]['event_length']['threshold_units'])
    del(example['sep_forecast_submission']['issue_time'])
    del(example['sep_forecast_submission']['forecasts'][0]['event_length'])
    del(example['sep_forecast_submission']['forecasts'][0]['threshold_crossings'])
    del(example['sep_forecast_submission']['forecasts'][1]['event_length'])
    del(example['sep_forecast_submission']['forecasts'][1]['threshold_crossings'])

#this is my sepmod template – may want to save this as a txt file if need to run this a lot of times
    sepmod_json = example

#data = example['sep_forecast_submission']['forecasts']
    data={}

#not 100% certain this is the best way to format it – will figure something out
#just put in same format as MAG4 for now
    for i in range(1,len(sepmod_data)):
        data[i-1]=sepmod_data[i]

#print(data)



#possibly need to add timezone to this
    for j in range(0,len(data)):
        data[j]['start_time'] = datetime.strptime(data[j]['start_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['start_time'] = data[j]['start_time'].isoformat()
        data[j]['end_time'] = datetime.strptime(data[j]['end_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['end_time'] = data[j]['end_time'].isoformat()
        data[j]['peak_time'] = datetime.strptime(data[j]['peak_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['peak_time'] = data[j]['peak_time'].isoformat()

    for i in range(len(data)):
        sepmod_json['sep_forecast_submission']['forecasts'][i]['energy_channel']['min'] = int(data[i]['energy_threshold'][1:])
        print('event energy threshold: %s' %sepmod_json['sep_forecast_submission']['forecasts'][i]['energy_channel']['min'])
        sepmod_json['sep_forecast_submission']['forecasts'][i]['all_clear']['threshold'] = data[i]['flux_threshold']
        sepmod_json['sep_forecast_submission']['forecasts'][i]['prediction_window']['start_time']=data[i]['start_time']
        sepmod_json['sep_forecast_submission']['forecasts'][i]['prediction_window']['end_time']=data[i]['end_time']
        sepmod_json['sep_forecast_submission']['forecasts'][i]['peak_intensity']['intensity']=data[i]['intensity']
        sepmod_json['sep_forecast_submission']['forecasts'][i]['peak_intensity']['time']=data[i]['peak_time']
        sepmod_json['sep_forecast_submission']['forecasts'][i]['fluence'][0]['fluence_value']=data[i]['fluence>10']
        sepmod_json['sep_forecast_submission']['forecasts'][i]['fluence'][1]['fluence_value']=data[i]['fluence>100']

        #could change this to be the threshold in the actual dictionary
        if (float(sepmod_json['sep_forecast_submission']['forecasts'][i]['peak_intensity']['intensity']) > float(sepmod_json['sep_forecast_submission']['forecasts'][i]['all_clear']['threshold'])):
            sepmod_json['sep_forecast_submission']['forecasts'][i]['all_clear']['all_clear_boolean'] = 'false'
        else:
            sepmod_json['sep_forecast_submission']['forecasts'][i]['all_clear']['all_clear_boolean'] = 'true'



    with open(model_path / output_file, 'w') as s:
        js.dump(sepmod_json,s,indent=1)


    #return output_file


#csv_files = [f for f in glob.glob("output/**.csv", recursive=True)]

#sepmod_files = [f for f in csv_files: if 'user' in str(f)]



def obs_csv2json(input_file,output_file,example_path):

    obs_path = Path(cfg.obs_path)

#pprint.pprint(sepmod_data)

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
    #print(example)
    example['sep_forecast_submission']['mode'] = 'observation'


#adding in things
    fluence = []
    fluence.append({'energy_min' : '10','fluence_value' : 'fluence_value', 'units' : 'MeV [cm^-2]'})
    fluence.append({'energy_min' : '100', 'fluence_value' : 'fluence_value', 'units' : 'MeV [cm^-2]'})

    ongoing_events = { "start_time": "2017-09-10T19:30Z", "threshold": 10, "energy_min": 10, "energy_max": -1 }

#this is my observation template – may want to save this as a txt file if need to run this a lot of times
    obs_json = example

    fieldnames = ('energy_threshold','flux_threshold','start_time','intensity','peak_time','rise_time','end_time','duration','fluence>10','fluence>100')

    with open(input_file,'r') as f:
        reader = csv.DictReader(f, fieldnames)
        out = js.dumps( [ row for row in reader ] )

    obs_data = js.loads(out)

#data = example['sep_forecast_submission']['forecasts']
    data={}

    obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['observatory'] = 'GOES-13'
#not 100% certain this is the best way to format it – will figure something out
#just put in same format as MAG4 for now
    for i in range(1,len(obs_data)):
        data[i-1]=obs_data[i]

    for j in range(0,len(data)):
        data[j]['start_time'] = datetime.strptime(data[j]['start_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['start_time'] = data[j]['start_time'].isoformat()
        data[j]['end_time'] = datetime.strptime(data[j]['end_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['end_time'] = data[j]['end_time'].isoformat()
        data[j]['peak_time'] = datetime.strptime(data[j]['peak_time'],'%Y-%m-%d %H:%M:%S')
        data[j]['peak_time'] = data[j]['peak_time'].isoformat()

    for i in range(len(data)):

        if i > 0:
          obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'].append(ongoing_events)

        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['energy_min'] = int(data[i]['energy_threshold'][1:])
        #print('event energy threshold: %s' %obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['energy_min'])
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['threshold'] = data[i]['flux_threshold']
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['start_time']=data[i]['start_time']
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['end_time']=data[i]['end_time']
        #this one doesnt have an option
        #obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events']['end_time']=data[0]['end_time']

        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['peak_intensity']=data[i]['intensity']
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['peak_time'] = data[i]['peak_time']
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['intensity_units']='pfu'
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['fluence'] = fluence
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['fluence'][0]['fluence']=data[i]['fluence>10']
        obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['fluence'][1]['fluence']=data[i]['fluence>100']

        if(float(obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['peak_intensity']) > float(obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['threshold'])):
            obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['all_clear_boolean'] = 'false'

        else:
            obs_json['sep_forecast_submission']['triggers'][0]['particle_intensity']['ongoing_events'][i]['all_clear_boolean'] = 'true'


    with open(obs_path / output_file, 'w') as s:
       js.dump(obs_json,s,indent=1)

    #return output_file

#SEPMOD_input_file = 'output/sep_values_user_integral_2012_1_23.csv'
#SEPMOD_output_file = 'sepmod_json_output.json'
#sepmod_csv2json(SEPMOD_input_file,SEPMOD_output_file)

#GOES_input_file = 'output\sep_values_GOES-13_integral_2012_1_27.csv'
#GOES_output_file = 'GOES-13_json_output.json'
#obs_csv2json(GOES_input_file,GOES_output_file,'example_sepscoreboard_json_file_v20190228.json')


