import json 
import os
import pprint as pp
import random

# file_name = '/home/ruff/work/hse/course:hough_analysis/code/markup.json'
file_name = 'markup.json'

file_names = []
with open(file_name) as f:
    data = json.load(f)
    for d in data:
        file_names.append(d)
    
    random.seed(9)
    test = random.sample(file_names, 100)
    tmp = list(set(file_names) - set(test))

    val = random.sample(tmp, 100)
    train = list(set(tmp) - set(val))

    val_json = json.dumps({v : data[v] for v in val})
    train_json = json.dumps({v : data[v] for v in train})
    test_json = json.dumps({v : data[v] for v in test})

    with open("val.json", "w") as f: 
        f.write(val_json) 

    with open("train.json", "w") as f: 
        f.write(train_json) 

    with open("test.json", "w") as f: 
        f.write(test_json) 
