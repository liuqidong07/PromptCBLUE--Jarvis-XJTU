# -*- encoding: utf-8 -*-
'''
@File    :   data_augment.py
@Time    :   2023/06/19 20:03:32
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import os
import random
import copy
import re
import json
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm


key_columns = ["input", "target", "answer_choices", "task_type", "task_dataset", "sample_id"]
file_path = "datasets/pre_data/"


def read_data(data_path):
    '''read data from jsonlines file'''
    data = []

    with jsonlines.open(file_path + data_path, "r") as f:
        for meta_data in f:
            data.append(meta_data)

    return data


def save_data(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open(file_path + data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)


def merge_data(data_list, data_path):
    '''Merge the raw and augment data'''
    all_data = []

    # merge data
    for meta_list in data_list:
        all_data += meta_list

    # save all of data
    print("Save the data to %s ... " % data_path)
    with jsonlines.open(file_path + data_path, "w") as w:
        for meta_data in tqdm(all_data):
            w.write(meta_data)


def aug_data(raw_data, templates, temp_re):
    '''Augment the data based on raw data'''
    new_data = []
    re1, re2, re3 = temp_re

    for meta_data in tqdm(raw_data):

        task = meta_data['task_dataset']

        flag = True
        # use re to extract the raw input query
        for index, temp in enumerate(re1[task]):

            if re.match(temp, meta_data['input']):  # find the current template
                
                current_temp = templates[task][index]
                current_temp_index = index
                flag = False
                break
        
        if flag:
            continue

        while 1:    # random sample a new template
            new_temp = random.choice(templates[task])
            if new_temp != current_temp:  # avoid use same template
                break

        if task in ["CMeEE-V2", "CMeIE", "CHIP-CDN", "CHIP-CTC", "KUAKE-QIC", 
                       "IMCS-V2-DAC", "IMCS-V2-NER", "IMCS-V2-SR"]:
        #if meta_data['task_type'] in ['cls', 'matching', 'ner', 'normalization']:
            raw_input = re.match(re1[task][current_temp_index], meta_data['input'])[1]
            new_input = copy.deepcopy(new_temp)
            new_input = new_input.replace("[INPUT_TEXT]", raw_input)
        
            option_list = copy.deepcopy(meta_data['answer_choices'])
            new_option_str = ''
            random.shuffle(option_list) # shuffle the raw option list
            for i, opt in enumerate(option_list):
                new_option_str += str(opt)

                if i < (len(option_list) - 1):
                    new_option_str += '，'
            
            #new_input = new_input.replace(option_str, new_option_str)
            new_input = new_input.replace("[LIST_LABELS]", new_option_str)

        elif task in ["CHIP-CDEE", "IMCS-V2-MRG", "MedDG"]:
            raw_input = re.match(re1[task][current_temp_index], meta_data['input'])[1]
            new_input = copy.deepcopy(new_temp)
            new_input = new_input.replace("[INPUT_TEXT]", raw_input)
        
        elif task in ["CHIP-STS", "KUAKE-QQR", "KUAKE-QTR"]:
            input_1 = re.match(re1[task][current_temp_index], meta_data['input'])[1]
            input_2 = re.match(re2[task][current_temp_index], meta_data['input'])[1]
            new_input = copy.deepcopy(new_temp)
            new_input = new_input.replace("[INPUT_TEXT_1]", input_2)
            new_input = new_input.replace("[INPUT_TEXT_2]", input_1)
        
        elif task in ["KUAKE-IR"]:
            input_1 = re.match(re1[task][current_temp_index], meta_data['input'])[1]
            input_2 = re.match(re2[task][current_temp_index], meta_data['input'])[1]
            new_input = copy.deepcopy(new_temp)
            new_input = new_input.replace("[INPUT_TEXT_1]", input_2)
            new_input = new_input.replace("[INPUT_TEXT_2]", input_1)
            option_list = copy.deepcopy(meta_data['answer_choices'])
            new_option_str = option_list[1] + "，" + option_list[0]
            new_input = new_input.replace("[LIST_LABELS]", new_option_str)

        elif task in ["CHIP-MDCFNPC"]:
            raw_input = re.match(re1[task][current_temp_index], meta_data['input'])[1]
            raw_mention = re.match(re3[task][current_temp_index], meta_data['input'])[1]
            new_input = copy.deepcopy(new_temp)
            new_input = new_input.replace("[INPUT_TEXT]", raw_input)
            new_input = new_input.replace("[LIST_MENTIONS]", raw_mention)

            option_list = copy.deepcopy(meta_data['answer_choices'])
            new_option_str = ''
            random.shuffle(option_list) # shuffle the raw option list
            for i, opt in enumerate(option_list):
                new_option_str += str(opt)

                if i < (len(option_list) - 1):
                    new_option_str += '，'
            
            new_input = new_input.replace("[LIST_LABELS]", new_option_str)

        new_meta_data = {key: meta_data[key] for key in key_columns}
        new_meta_data["input"] = new_input
        if task in ["CMeEE-V2", "CMeIE", "CHIP-CDN", "CHIP-CTC", "KUAKE-QIC", 
                       "IMCS-V2-DAC", "IMCS-V2-NER", "IMCS-V2-SR", "CHIP-MDCFNPC"]:
            new_meta_data["answer_choices"] = option_list
        new_data.append(new_meta_data)

    print("The number of augmentation is %d." % len(new_data))
        
    return new_data


def get_re(templates):
    '''get the re formula for input and labels'''
    re1, re2, re3 = {}, {}, {}

    for key, value in templates.items():

        re1[key] = []
        re2[key] = []
        re3[key] = []

        for temp in value:

            if key in ["CMeEE-V2", "CMeIE", "CHIP-CDN", "CHIP-CTC", "KUAKE-QIC", 
                       "IMCS-V2-DAC", "IMCS-V2-NER", "IMCS-V2-SR"]:
                re1[key].append(temp.replace("[INPUT_TEXT]", "([\s\S]*)").replace("[LIST_LABELS]", "[\s\S]*").replace("\\n答：", ""))   # match and ignore
                re2[key].append(temp.replace("[LIST_LABELS]", "([\s\S]*)").replace("[INPUT_TEXT]", "[\s\S]*").replace("\\n答：", ""))
            elif key in ["CHIP-CDEE", "IMCS-V2-MRG", "MedDG"]:
                re1[key].append(temp.replace("[INPUT_TEXT]", "([\s\S]*)").replace("\\n答：", ""))
            elif key in ["CHIP-STS", "KUAKE-QTR"]:
                re1[key].append(temp.replace("[INPUT_TEXT_1]", "([\s\S]*)").replace("[INPUT_TEXT_2]", "[\s\S]*").replace("\\n答：", ""))
                re2[key].append(temp.replace("[INPUT_TEXT_2]", "([\s\S]*)").replace("[INPUT_TEXT_1]", "[\s\S]*").replace("\\n答：", ""))
            elif key in ["KUAKE-QQR", "KUAKE-IR"]:
                re1[key].append(temp.replace("[INPUT_TEXT_1]", "([\s\S]*)").replace("[INPUT_TEXT_2]", "[\s\S]*").replace("[LIST_LABELS]", "[\s\S]*").replace("\\n答：", ""))
                re2[key].append(temp.replace("[INPUT_TEXT_2]", "([\s\S]*)").replace("[INPUT_TEXT_1]", "[\s\S]*").replace("[LIST_LABELS]", "[\s\S]*").replace("\\n答：", ""))
                #TODO:lack the LIST_LABELS
            elif key in ["CHIP-MDCFNPC"]:
                re1[key].append(temp.replace("[INPUT_TEXT]", "([\s\S]*)").replace("[LIST_LABELS]", "[\s\S]*").replace("[LIST_MENTIONS]", "[\s\S]*").replace("\\n答：", ""))
                re2[key].append(temp.replace("[LIST_LABELS]", "([\s\S]*)").replace("[INPUT_TEXT]", "[\s\S]*").replace("[LIST_MENTIONS]", "[\s\S]*").replace("\\n答：", ""))
                re3[key].append(temp.replace("[LIST_MENTIONS]", "([\s\S]*)").replace("[INPUT_TEXT]", "[\s\S]*").replace("[LIST_LABELS]", "[\s\S]*").replace("\\n答：", ""))
            else:
                raise ValueError

    return re1, re2, re3


def add_choice_CTC_QIC(data):
    '''add 非上述类型 to task CHIP-CTC and KUAKE-QIC. Though IMCS-V2-DAC has the same problem,
      but it is always 100.'''
    for meta_data in data:

        task = meta_data["task_dataset"]

        if (task == "CHIP-CTC") | (task == "KUAKE-QIC"):

            option_list = meta_data["answer_choices"]
            raw_option_str = ''

            for i, opt in enumerate(option_list):
                raw_option_str += str(opt)
                if i < (len(option_list) - 1):
                    raw_option_str += '，'
            
            new_option_str = raw_option_str + '，非上述类型'

            meta_data["input"] = meta_data["input"].replace(raw_option_str, new_option_str)
            meta_data["answer_choices"].append("非上述类型")


if __name__ == '__main__':

    task_path = "partition/KUAKE-IR"
    train_path = os.path.join(task_path, "train.json")
    dev_path = os.path.join(task_path, "dev.json")
    test_path = os.path.join(task_path, "test.json")
    index_aug = 2
    aug_num = 1

    raw_data = read_data(train_path)
    #add_choice_CTC_QIC(raw_data)
    templates = json.load(open(os.path.join(file_path, "templates/templates.json"), "r"))
    temp_re = get_re(templates)

    augmentation = []
    for _ in range(aug_num):
        augmentation.append(aug_data(raw_data, templates, temp_re))

    merge_data([raw_data] + augmentation, os.path.join(task_path, "train_aug.json"))

    # dev_data = read_data(dev_path)
    # add_choice_CTC_QIC(dev_data)
    # save_data("dev_CQ.json", dev_data)

    # test_data = read_data(test_path)
    # add_choice_CTC_QIC(test_data)
    # save_data("test_CQ.json", test_data)




