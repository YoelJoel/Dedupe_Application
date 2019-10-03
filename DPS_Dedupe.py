# Setup
import pandas as pd
import numpy as np
from future.builtins import next
import os
import csv
import re
import logging
import optparse
import dedupe
from unidecode import unidecode


df = pd.read_csv("",header=0,index_col=False)
df1 = pd.DataFrame(df)

# Find postcodes which appear more than 3 times, to reduce data

sub_df = df1[['Property Postcode','Property Address']]
df_sum = sub_df.groupby('Property Postcode')\
    ['Property Address'].count()\
    .reset_index(name="count")
df_sum1 = df_sum[df_sum['count'] >= 3].reset_index()
df_sum2 =  df_sum1['Property Postcode']
red_DPSTDD = df1[~df1['Property Postcode'].isin(df_sum2) == False].reset_index()
red_DPSTDD.to_csv(r'reduced data')

input_file = "reduced data"
output_file = "cluster data"
settings_file = 'settings file'
training_file = "labelled pairs"

# ## Setup



def NormalizeAddress(addressline):
    """
    
    """
    addressline = re.sub(',',' ', addressline)
    addressline = re.sub('/',' ', addressline)
    addressline = re.sub('[^A-Za-z0-9]+',' ', addressline)
    addressline = re.sub('\[',' ', addressline)
    addressline = re.sub('\]',' ', addressline)
    addressline = re.sub('\(|\)',' ', addressline)
    addressline = re.sub('\[[^]]*\]',' ', addressline)
    addressline = re.sub('  ',' ', addressline)
    addressline = re.sub('\n',' ', addressline)
    addressline = addressline.strip().strip('"').strip("'")\
        .lower().strip()
    if not addressline:
        addressline = None
    return addressline





def readData(filename):
    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, NormalizeAddress(v)) for (k, v) in row.items()]
            row_id = int(row[''])
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

Clustered_Filtered = pd.read_csv("clustered data")
Filtered_Sum = Clustered_Filtered.groupby('Cluster ID')\
    ['Property Address'].count()\
    .reset_index(name="count")
Filtered_Sum1 = Filtered_Sum[Filtered_Sum['count'] >= 3].reset_index()
print(Filtered_Sum1)
Filtered_Sum2 = Filtered_Sum1['Cluster ID']
DPSTDD_Final = Clustered_Filtered[~Clustered_Filtered['Cluster ID'].isin(Filtered_Sum2) == False].reset_index()
DPSTDD_Final.to_csv(r'clean clustered data')
