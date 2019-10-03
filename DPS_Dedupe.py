# Setup

import pandas as pd
import numpy as np

#string matching
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process

from future.builtins import next

import os
import csv
import re
import logging
import optparse

import dedupe
from unidecode import unidecode



# Importing Data

# df = pd.read_csv("D:\MSc Project\Data\DPS_Deposits\DPSTDD.csv",header=0,index_col=False)
# df1 = pd.DataFrame(df)

# Find postcodes which appear more than 3 times, to reduce data

# sub_df = df1[['Property Postcode','Property Address']]
# df_sum = sub_df.groupby('Property Postcode')\
#     ['Property Address'].count()\
#     .reset_index(name="count")
# df_sum1 = df_sum[df_sum['count'] >= 3].reset_index()
# print(df_sum1['count'].sum()) # data reduced from 4995 to 3175 instances
# df_sum2 =  df_sum1['Property Postcode']
# red_DPSTDD = df1[~df1['Property Postcode'].isin(df_sum2) == False].reset_index()
# print(red_DPSTDD)
# red_DPSTDD.to_csv(r'D:\MSc Project\Data\DPS_Deposits\DPSTDD_Clean.csv')
# # Remove addresses which aren't in Bedfordshire
# #BedsCodes = ('LU1','LU3','LU4','LU5','LU6','LU7','MK17','MK40','MK41','MK42',
# #            'MK43','MK44','MK45','NN10','NN29','PE18','PE19','SG15','SG16',
# #           'SG17','SG18','SG19','SG5')
#
#
#
input_file = "D:\\MSc Project\\Data\\DPS_Deposits\\DPSTDD_Clean.csv"
output_file = "D:\\MSc Project\\Data\\DPS_Deposits\\DPSTDD_Clustered.csv"
settings_file = 'DPSTDD_Predicate_Settings2'
training_file = "D:\\MSc Project\\Data\\DPS_Deposits\\DPSTDD_Training5.json"

# ## Setup



def preProcess(column):
    try:
        column = column.decode('utf8')
    except AttributeError:
        pass
    column = unidecode(column)
    column = re.sub('/',' ', column)
    column = re.sub('-', ' ', column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    if not column:
        column = None
    return column


def readData(filename):


    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = int(row['newindex'])
            data_d[row_id] = dict(clean_row)

    return data_d


print('importing data ...')
data_d = readData(input_file)

if os.path.exists(settings_file):
    print('reading from', settings_file)
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticDedupe(f)
else:
    fields = [
        {'field': 'Property Address', 'type': 'String'},
        {'field': 'Property Postcode', 'type': 'String'},
    ]

#        {'field': 'Rental Property Line2', 'type': 'String'},
#'field': 'Rental Property Line3', 'type': 'String', 'has missing': True


    deduper = dedupe.Dedupe(fields)

    deduper.sample(data_d, 15000)

    if os.path.exists(training_file):
        print('reading labeled examples from ', training_file)
        with open(training_file, 'rb') as f:
            deduper.readTraining(f)

    print('starting active labeling...')

    dedupe.consoleLabel(deduper)

    deduper.train(recall=0.5)

    with open(training_file, 'w') as tf:
        deduper.writeTraining(tf)

    with open(settings_file, 'wb') as sf:
        deduper.writeSettings(sf)


threshold = deduper.threshold(data_d, recall_weight=0.5)


print('clustering...')
clustered_dupes = deduper.match(data_d, threshold)

print('# duplicate sets', len(clustered_dupes))


# Write our original data back out to a CSV with a new column called
# 'Cluster ID' which indicates which records refer to each other.

cluster_membership = {}
cluster_id = 0
for (cluster_id, cluster) in enumerate(clustered_dupes):
    id_set, scores = cluster
    cluster_d = [data_d[c] for c in id_set]
    canonical_rep = dedupe.canonicalize(cluster_d)
    for record_id, score in zip(id_set, scores):
        cluster_membership[record_id] = {
            "cluster id": cluster_id,
            "canonical representation": canonical_rep,
            "confidence": score
        }

singleton_id = cluster_id + 1

with open(output_file, 'w') as f_output, open(input_file) as f_input:
    writer = csv.writer(f_output)
    reader = csv.reader(f_input)

    heading_row = next(reader)
    heading_row.insert(0, 'confidence_score')
    heading_row.insert(0, 'Cluster ID')
    canonical_keys = canonical_rep.keys()
    for key in canonical_keys:
        heading_row.append('canonical_' + key)

    writer.writerow(heading_row)

    for row in reader:
        row_id = int(row[0])
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]["cluster id"]
            canonical_rep = cluster_membership[row_id]["canonical representation"]
            row.insert(0, cluster_membership[row_id]['confidence'])
            row.insert(0, cluster_id)
            for key in canonical_keys:
                row.append(canonical_rep[key].encode('utf8'))
        else:
            row.insert(0, None)
            row.insert(0, singleton_id)
            singleton_id += 1
            for key in canonical_keys:
                row.append(None)
        writer.writerow(row)

Clustered_Filtered = pd.read_csv("D:\\MSc Project\\Data\\DPS_Deposits\\DPSTDD_Clustered.csv")
Filtered_Sum = Clustered_Filtered.groupby('Cluster ID')\
    ['Property Address'].count()\
    .reset_index(name="count")
Filtered_Sum1 = Filtered_Sum[Filtered_Sum['count'] >= 3].reset_index()
print(Filtered_Sum1)
Filtered_Sum2 = Filtered_Sum1['Cluster ID']
DPSTDD_Final = Clustered_Filtered[~Clustered_Filtered['Cluster ID'].isin(Filtered_Sum2) == False].reset_index()
DPSTDD_Final.to_csv(r'D:\MSc Project\Data\DPS_Deposits\DPSTDD_Final.csv')
