import numpy as np
from tqdm import tqdm
from tqdm import trange
import math
import csv
import base64
from io import StringIO
import os
import time

contrast_dict = {'0.0':0, '0.5235987755982988':1, '1.0471975511965976':2,'1.5707963267948966':3,'2.0943951023931953':4,'2.6179938779914944':5,
 '3.141592653589793':6,'3.6651914291880923':7,'4.1887902047863905':8,'4.71238898038469':9,'5.235987755982989':10,'5.759586531581287':11}


def get_heading_degree():
    new_headings = []
    new_elevation = [30*math.pi/180, 0*math.pi/180, -30*math.pi/180]

    for i in range(0, 360, 30):
        current_radians = i*math.pi/180
        new_headings.append(current_radians)
    return new_headings, new_elevation



def get_view1(elevation, heading):
    if elevation < 0:
        key_index = contrast_dict[str(heading)]+0*12
    elif elevation == 0:
        key_index = contrast_dict[str(heading)]+1*12
    else:
        key_index = contrast_dict[str(heading)]+2*12
   
    return key_index

def get_view2():
    all_data = np.load('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/152-object_feature_new_v4.npy',allow_pickle=True).item()
    for scan in all_data.values():
        for viewpoint in tqdm(scan.values()):
            for key, value in viewpoint.items():
                if isinstance(key, int):
                    continue
                tmp_key = key.split('_')
                if "-" in tmp_key[-1]:
                    key_index = contrast_dict[tmp_key[0]]+0*12
                elif tmp_key[-1] == "0.0":
                    key_index = contrast_dict[tmp_key[0]]+1*12
                else:
                    key_index = contrast_dict[tmp_key[0]]+2*12
                viewpoint[key_index] = value
                viewpoint.pop(key)


relative_path = "/egr/research-hlr/joslin/Matterdata/v1/scans/pre-trained2/"
standard_heading, standard_elevation = get_heading_degree()

if __name__ == "__main__":  
    long_env = []
    with open("/VL/space/zhan1624/obj-vln/r2r_src/env_short.txt") as f1:
        for line in f1:
            long_env.append(line.strip())
    
    all_scenery_list = os.listdir(relative_path)
    all_scenery_list = [i[:-4] for i in all_scenery_list]
    all_scenery_list = long_env
    

    for scenry in tqdm(all_scenery_list): # each in 61 scans; scenery is the first index in 5
        all_data_dict = {}
        temp_feature = []# collect all 1 features
        new_scenry = np.load(relative_path + scenry + ".npy", allow_pickle=True).item()
        tmp_scenery = {}
        for each_state, value in new_scenry.items():
            tmp_feature_value = {}
            for each_elevation in standard_elevation:
                for each_heading in standard_heading:
                    tmp_feature_value[get_view1(each_elevation, each_heading)] = value[str(each_heading)+"_"+str(each_elevation)]['features']
            #tmp_scenery[each_state] = np.stack(list(zip(*sorted(tmp_feature_value.items())))[1],axis=0)
            tmp_scenery[each_state] = tmp_feature_value
        all_data_dict[scenry] = tmp_scenery
        np.save('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/short_env_objects/'+scenry, all_data_dict)
    #np.save('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/mycsvfile_image_text',all_data_dict)
    

    '''
    vocab_list = []
    dim_list = []xP
    vocab_list_token = {}
    old_dict = np.load('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/152-object_feature_new_v4.npy', allow_pickle=True).item()
    for scan_id, each_scan in tqdm(old_dict.items()):
        for state_id, each_state in each_scan.items():
            for each_id, each_heading in each_state.items():
                # each_heading.pop("features")
                # each_heading.pop("image_h")
                # each_heading.pop("image_w")
                # each_heading.pop("num_boxes")
                # new_text_list = []
                # for each_word in each_heading['text']:
                #     new_text_list.append(vocab_dict[each_word])
                # each_heading['text_feature'] = np.array(new_text_list)
                for word_id, each_word in enumerate(each_heading['text']):
                    if each_word not in vocab_list:
                        vocab_list.append(each_word)
                        dim_list.append(each_heading['text_feature'][word_id])
    for vocab_id, each_vocb in enumerate(vocab_list):
        tmp_dict = {}
        tmp_dict["id"] = vocab_id
        tmp_dict['dim'] = dim_list[vocab_id]
        vocab_list_token[each_vocb] = tmp_dict
    np.save('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/mycsvfile2',vocab_list_token)
    '''

    
    
    # long_env = []
    # with open("/VL/space/zhan1624/obj-vln/r2r_src/env_short.txt") as f1:
    #     for line in f1:
    #         long_env.append(line.strip())
    aa = np.load('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/mycsvfile_image_text.npy',allow_pickle=True).item()
    # new_dict= {}
    # for key, value in tqdm(aa.items()):
    #     if key in long_env:
    #         new_dict[key] = value
    np.savez('/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/new_mycsvfile_image_text_copy1',aa)
    print('yue')