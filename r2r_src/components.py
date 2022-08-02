import json
import numpy as np
import torch
import math
from itertools import chain
import spacy
from collections import defaultdict
# from itertools import chain



object_class = []
base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')
specific_word = ["after", "once", "until"]
config = {
    'motion_indicator_file' : 'tasks/R2R-pano/data/data/component_data/motion_indicator/motion_dict.txt',
    'split_file' : 'tasks/R2R-pano/data/data/split_dictionary.txt',
    'stop_words_file': 'tasks/R2R-pano/data/data/stop_words.txt',
    'train_file': 'tasks/R2R-pano/data/data/R2R_train.json',
    'position_file':"tasks/R2R-pano/data/data/spatial_position_dic.txt",
    'spatial_indicator_file':"tasks/R2R-pano/data/data/spatial_indicator.txt"
}
nlp = spacy.load("en_core_web_lg")

def split_oder(dictionary):
    return sorted(dictionary, key = lambda x: len(x.split()), reverse=True)

with open(config['split_file']) as f_dict:
    dictionary = f_dict.read().split('\n')
    dictionary = split_oder(dictionary)
    tmp_dict = list(filter(None,[each_phrase+"," for each_phrase in dictionary]))
    dictionary = dictionary + tmp_dict
    dictionary = [" "+each_phrase.strip()+" " for each_phrase in dictionary]

with open(config['motion_indicator_file']) as f_dict:
    motion_dict = f_dict.read().split('\n')
    motion_dict = [each_motion.strip() for each_motion in motion_dict]
    motion_dict = split_oder(motion_dict)

with open(config["stop_words_file"]) as f_stop_word:
    stopword = f_stop_word.read().split('\n')
    stopword = split_oder(stopword)


def read_file(file_path):
    with open(file_path) as f_dict:
        read_list = f_dict.read().split('\n')
        read_list = [each.strip() for each in read_list]
        read_list = split_oder(read_list)
    return read_list

position_list = read_file(config['position_file'])  

spatial_indicator_list = read_file(config['spatial_indicator_file'])
    

def get_configurations(sentence):
    sentence = sentence.lower().strip()
    sentence_list = sentence.split('.')
    sentence_list = [('', " "+each_sentence+" ") for each_sentence in sentence_list]
    for each_word in dictionary:
        for sl in list(sentence_list):
            if each_word in sl[1]:
                index = sentence_list.index(sl)
                sentence_list.remove(sl)
                tmp_word = sl[1].split(each_word)
                for id, tt in enumerate(tmp_word):
                    if id == 0:
                        tmp_tuple = (sl[0], " "+ tt + " ")
                    else:
                        tmp_tuple = (each_word, " "+ tt + " ")
                    sentence_list.insert(index, tmp_tuple)
                    index += 1
    
  
    sentence_list = combine_process(sentence_list)
    sentence_list = [' '.join(filter(None, map(str.strip, each_sentence))) for each_sentence in sentence_list]
    sentence_list = post_processing_sentence(list(filter(None,sentence_list)))
    return sentence_list



# def get_sentence_configuration(sentence):
#     print('a',sentence)
#     for each_word in dictionary:
#         sentence_list = sentence.split(each_word)
#         if len(sentence_list)==1:
#             continue
#         print('c',sentence_list)
#         sentence_list = list(map(get_sentence_configuration, sentence_list))
#         print('b',sentence_list)
#         assert sentence_list[0][0] == ''
#         sentence_list[0] = (each_word, sentence_list[0][1])
#         return sentence_list
#     else:
#         return ['', sentence]

# def get_configuration(sentence):
#     sentence = sentence.lower().strip()
#     sentence_list = sentence.split('.')
#     return list(map(get_sentence_configuration, sentence_list))


def post_processing_sentence(sentence_list):
    def func(sl):
        sl = sl.strip().strip(',').strip('.')
        for sw in stopword:
            if sl.endswith(" %s"%sw):
                sl = sl[:-(len(sw)+1)]
            elif sl.endswith("%s"%sw):
                sl = sl[:-len(sw)]
            elif sl.startswith("%s "%sw) or sl.startswith(" %s"%sw):
                sl = sl[len(sw)+1:]
            elif sl.startswith("%s"%sw):
                sl = sl[len(sw):]
        return sl
    sentence_list = list(map(func, sentence_list))
    new_sentence_list = []
    tmp_id = 100
    for id, sent in enumerate(sentence_list):
        if sent !='':
            sent = sent.strip().strip(',').strip('.')
            if sent.endswith('until you'):
                tmp_sent = sent + " " + sentence_list[id+1]
                tmp_id = id + 1
                new_sentence_list.append(tmp_sent)
            else:
                if id == tmp_id:
                    continue
                else:
                    new_sentence_list.append(sent.strip())
            
    return new_sentence_list

def combine_process(config_list):
    new_config_list = []
    tmp_config_list = []
    for id, each_config in enumerate(config_list):
        tmp_config_list.append(each_config)
        if each_config[0] != '' :
            new_config_list.append(tmp_config_list)
            tmp_config_list = []
    if tmp_config_list:
        if new_config_list:
            new_config_list[-1].extend(tmp_config_list)
        else:
            new_config_list.append(tmp_config_list)
        tmp_config_list = []

    
    new_config_list = [list(chain(*each_new_config)) for each_new_config in new_config_list]
    return new_config_list
    
            
def compare_config(gold_file, compare_file):
    def build_dict(config_list):
        config_dict = {}
        for each_config in config_list:
            each_config = each_config.strip('\n').strip().split('\n')
            config_dict[each_config[0]] = each_config[1:]
        return config_dict

    with open(gold_file) as f_gold, open(compare_file) as f_comp:
        gold_list = f_gold.read().split('\n\n')
        comp_list = f_comp.read().split('\n\n')
    gold_dict = build_dict(gold_list)
    comp_dict = build_dict(comp_list)
    count = 0
    for gold_key, gold_value in gold_dict.items():
        if comp_dict[gold_key][1:] == gold_value[1:]:
            continue
        else:
            count += 1
            print(gold_value)
            print(comp_dict[gold_key])
            print('\n')
    print(count, len(gold_dict))

def result_compare(path1, path2, path3):
    def get_heading(heading):
        temp = int(round((heading*180/math.pi)/30) * 30)
        if temp >=  360:
            temp = temp - 360      
        elif temp < 0 :
            temp = temp + 360
        return temp*math.pi/180
    
    def generate_pair(input_list):
        pair_list = []
        for id, element in enumerate(input_list):
            if id != len(input_list)-1:
                pair_list.append((element, input_list[id+1]))
            else:
                break
        return pair_list

    predict_goal_dict = {}
    gold_goal_dict = {}
    count = 0
    all_landmark_list = []
    object_vocab_list = []
    object_names_dict = np.load('img_features/object_name.npy', allow_pickle=True).item()
    print(object_names_dict['HxpKQynjfin']['7a920bf869fd41099e44a0829f818361'][get_heading(5.723959717784949)]['text'])
    
    # with open(path1) as f_comp1, open(path2) as f_comp2, open(path3) as f_comp3:
    #     predicate_list = json.load(f_comp1)
    #     gold_list = json.load(f_comp2)
    #     landmark_dict = json.load(f_comp3)
    
    # for landmark_key, landmark_value in landmark_dict.items():
    #     all_landmark_list += landmark_value
    
    # all_landmark_list = list(set(all_landmark_list))
    # with open('/VL/space/zhan1624/selfmonitoring-agent/img_features/fast_Rcnn_object_vocab.txt') as f_vocb:
    #     for line in f_vocb:
    #         line = line.strip()
    #         if "," in line:
    #             line = line.split(',')
    #             object_vocab_list += line
    #         else:
    #             object_vocab_list.append(line)
    
    # for each_all_landmark in all_landmark_list:
    #     if each_all_landmark in object_vocab_list:
    #         count+=1
    # print(count)


    # for each_pre in predicate_list:
    #     predict_goal_dict[each_pre['instr_id']] = each_pre
    # for each_gold in gold_list:
    #     gold_goal_dict[each_gold['path_id']] = each_gold
    
    # predict_goal_list = list(predict_goal_dict.items())
    # for each_pre_key, each_pre_value in predict_goal_list:
    #     pred_trajectory = list(list(zip(*each_pre_value['trajectory']))[0])
    
    #     tmp_pre_key = each_pre_key.split('_')[0]
    #     each_gold_goal = gold_goal_dict[int(tmp_pre_key)]['path']
        
    #     gold_pair_list = generate_pair(each_gold_goal)
    #     pred_pair_list = generate_pair(pred_trajectory)

    #     for pred_id, pred_pair in enumerate(pred_pair_list):
    #         if pred_pair in gold_pair_list:
    #             if gold_pair_list.index(pred_pair) == 5:
    #                 count+=1
    #                 break
                 

    #         tmp_list = []
    #         scan_id = gold_goal_dict[int(tmp_pre_key)]['scan']
    #         for each_traj in each_pre_value['trajectory']:
    #             each_heading = get_heading(each_traj[1])
    #             each_landmark= object_names_dict[scan_id][each_traj[0]][each_heading]['text']
    #             new_each_traj[each_traj[0]] = sorted(set(each_landmark),key=each_landmark.index)
    #             tmp_list += new_each_traj[each_traj[0]]
            
    #         for extract_landmark in landmark_dict[each_pre_value['instr_id']]:
    #             if extract_landmark not in tmp_list:
    #                 count += 1
    #                 break
    #         total_num +=1
    # print(total_num)
    # print(count)
    # print(count/total_num)

        

def get_motion_indicator1(test_sentence, split_dict):
    motion_indicator = ""
    for each_word in split_dict:
        if each_word in test_sentence:
            motion_indicator = each_word

    if motion_indicator == '':
        doc = nlp(test_sentence)
        for token_id, token in enumerate(doc):
            try:
                if token.pos_ == "VERB" and doc[token_id+1].pos_ == "ADP":
                    motion_indicator = token.text + " " + doc[token_id+1].text
                elif token.pos_ == "VERB":
                    motion_indicator = token.text 
                else:
                    motion_indicator = doc[0].text    
            except IndexError:
                    motion_indicator = token.text
    
    motion_doc = nlp(motion_indicator)
    sum_list = []
    for motion_token in motion_doc:
        sum_list.append(motion_token.vector)
    motion_indicator_vector = np.mean(np.array(sum_list), axis=0)
        
    return motion_indicator, motion_indicator_vector




    
    # motion_doc = nlp(motion_indicator)
    # sum_list = []
    # for motion_token in motion_doc:
    #     sum_list.append(motion_token.vector)
    # motion_indicator_vector = np.mean(np.array(sum_list), axis=0)



def get_landmark(test_sentence, split_dict):
    doc = nlp(test_sentence)
    landmark_stopwords = ['right', 'left','front','them', 'you','end','top', 'bottom','it','middle','side']
    landmark_list =[]
    if test_sentence in split_dict:
        landmark_list.append(('', np.zeros(300)))
        return landmark_list

    for chunk in doc.noun_chunks:
        landmark_text = chunk.root.text
        landmark_vector = chunk.root.vector
        if landmark_text not in landmark_stopwords:
            landmark_list.append((landmark_text, landmark_vector))
    return landmark_list

postion_list = read_file(config['position_file'])  

spatial_indicator_list = read_file(config['spatial_indicator_file'])


def get_motion_dict(test_sentence):
    motion_indicaotr_dict = {}
    doc = nlp(test_sentence) 
    window = 5    
    motion_indicator = ""
    for motion_token_id, motion_token in enumerate(doc):
        if motion_token.pos_ == "VERB":
            motion_indicaotr_dict['motion'] = motion_token.text
            left_tokens = list(reversed(doc[motion_token_id+1:motion_token_id + window]))
            for left_id, left_token in enumerate(left_tokens):
                if left_token.text in postion_list:
                    motion_indicaotr_dict['direction'] = left_token.text
                    motion_indicator = " ".join([each_m.text for each_m in  doc[motion_token_id:len(left_tokens)-left_id +1]])
                    motion_indicaotr_dict['motion_indcator'] = motion_indicator
            if motion_indicator == '':
                motion_indicator = doc[0].text
            break

    return motion_indicator

def get_motion_indicator2(test_sentence, split_dict):
    motion_indicator = ""
    motion_indicator_dict = {}
    doc = nlp(test_sentence)
    window = 5

    for motion_token_id, motion_token in enumerate(doc):  
        if motion_token.pos_ == "VERB":
            tmp_motion_token = motion_token.text
            left_tokens = list(reversed(doc[motion_token_id+1:motion_token_id + window]))
            for left_id, left_token in enumerate(left_tokens):
                if left_token.text in postion_list:
                    last_token_id = len(left_tokens)-left_id +1
                    tmp_position_token = left_token.text
                    motion_indicator = " ".join([each_m.text for each_m in doc[motion_token_id:len(left_tokens)-left_id +1]])
                    tmp_motion_indicator = motion_indicator
            if motion_indicator in motion_dict:
                motion_indicator_dict['motion'] = tmp_motion_token
                motion_indicator_dict['direction'] = tmp_position_token
                motion_indicator_dict['motion_indcator'] = tmp_motion_indicator
                try:
                    if doc[last_token_id].pos_ == "ADP":
                        motion_indicator_dict['spatial_indicator'] = doc[last_token_id].text
                except IndexError:
                    print("quan")
            else:
                if left_tokens[-1].pos_ == 'ADP':
                    motion_indicator = motion_token.text + " " + left_tokens[-1].text
                    motion_indicator_dict['motion'] = motion_token.text
                    motion_indicator_dict['motion_indcator'] = motion_indicator
                    motion_indicator_dict['spatial_indicator'] = left_tokens[-1].text
                else:
                    motion_indicator = motion_token.text
                    motion_indicator_dict['motion_indcator'] = motion_indicator          

            if motion_indicator == '':
                motion_indicator = doc[0].text
            break

    return motion_indicator_dict

def check_contain(check_word, check_dictionary, contain_stop=False):
    check_word = check_word.strip()
    if contain_stop:
        connect_not_list = ['and', 'and then', 'or']
        for e_not in connect_not_list:
            if e_not in check_word:
                return False
    for each_c_d in check_dictionary:
        if each_c_d in check_word:
            return True


def get_landmark2(test_sentence, split_dict): # "and" “of” are not processed yet
    #test_sentence = "wait on the first landing by the banister"
    #test_sentence = "go down the stairs to the bottom of the first landing"
    #test_sentence = "stop by the latticework inside the house directly in front of the door"
    #test_sentence = "stand in front of the dining table on the bench side"
    #test_sentence = "walk through doorway just right of the portraits of a family"
    #test_sentence = "head straight until you pass the wall with holes in it"
    #test_sentence = "wait by the glass table with the white chairs"
    #test_sentence = "stopping at the picture with prayer on it"
    #test_sentence = "enter the house by the larger doorway with the two glass doors"
    #test_sentence = "walk passed the sink and stove area"
    #test_sentence = "stop between the refrigerator and dining table"
    #test_sentence = "walk past the bench and then past the island"
    doc = nlp(test_sentence)
    window = 5
    start_id = 0
    start_end_id = 0
    end_id = 0
    landmark_stopwords = ['right', 'left','front','them', 'you','end','top', 'bottom','it','middle','side']
    connect_list = ['with','by','of','on']
   
    landmark_list =[]
    new_none_chunks = []
    if test_sentence in motion_dict:
        #landmark_list.append(('', np.zeros(300)))
        landmark_list = []
        return landmark_list
    else:
        #check whether two landmarks could be combined together
        noun_chunks = list(doc.noun_chunks)
        noun_chunks = [each_chunk for each_chunk in noun_chunks if each_chunk.text not in landmark_stopwords]
        for e_c_id, each_chunk in enumerate(noun_chunks):  
            between_token_list = []
            
            for e_t_id, each_token in enumerate(doc):
                if doc[e_t_id:e_t_id+1].start_char == each_chunk.start_char: #begin start
                    start_id = e_t_id

                if doc[e_t_id:e_t_id+1].end_char == each_chunk.end_char: #begin end
                    start_end_id = e_t_id

                if e_c_id + 1 < len(noun_chunks):
                    if doc[e_t_id:e_t_id+1].end_char == noun_chunks[e_c_id+1].end_char:# next end
                        end_id = e_t_id

                    if doc[e_t_id:e_t_id+1].end_char > each_chunk.end_char and doc[e_t_id:e_t_id+1].start_char < noun_chunks[e_c_id+1].start_char:
                        between_token_list.append(each_token.text)
                        
                        
                # to process the last chunk
                else:
                    if not new_none_chunks:
                        if start_id and start_end_id:
                            new_none_chunks.append(((e_c_id, start_id, each_chunk), None, (e_c_id, start_end_id, each_chunk)))
                            break
                    else:
                        if e_c_id != new_none_chunks[-1][-1][0]:
                            if start_id and start_end_id:
                                new_none_chunks.append(((e_c_id, start_id, each_chunk), None, (e_c_id, start_end_id, each_chunk)))
                                break

            if between_token_list:
                connection_phrase = " ".join(between_token_list) 

                #for the landmarks that are connected with the spatial indicators
                if check_contain(connection_phrase, connect_list, contain_stop=True) or check_contain(connection_phrase, position_list, contain_stop=True):
                    #for each landmark: landmark_id, start_index, landmark_text
                    new_none_chunks.append(((e_c_id, start_id, each_chunk), connection_phrase, (e_c_id+1, end_id, noun_chunks[e_c_id+1])))
                    
                else:
                    if not new_none_chunks:
                        if start_id and start_end_id:
                            new_none_chunks.append(((e_c_id, start_id, each_chunk), None, (e_c_id, start_end_id, each_chunk)))
                            
                    elif e_c_id == new_none_chunks[-1][-1][0]:
                        continue
                    else:
                        if start_id and start_end_id:
                            new_none_chunks.append(((e_c_id, start_id, each_chunk), None, (e_c_id, start_end_id, each_chunk)))
                        

            

        # spatial indicator
        tmp_new_none_chunks = []
        tmp_chunk = []
        for l_r_id, landmark_tuple in enumerate(new_none_chunks):
            if not tmp_chunk:
                tmp_chunk.append(landmark_tuple)
            else:
                if landmark_tuple[0][0] == tmp_chunk[-1][-1][0]:
                    tmp_chunk.append(landmark_tuple) 
                else:      
                    tmp_new_none_chunks.append(tmp_chunk)
                    tmp_chunk = []
                    tmp_chunk.append(landmark_tuple)
            if l_r_id == len(new_none_chunks)-1:
                tmp_new_none_chunks.append(tmp_chunk)

        # obtain spatial indicator
        final_none_chunks = []
        for e_l_l, landmark_list in enumerate(tmp_new_none_chunks):
            tmp_spa_list = []
            landmark_span = doc[landmark_list[0][0][1]:landmark_list[-1][-1][1]+1]
            for e_t_id, each_token in enumerate(doc):
                if doc[e_t_id:e_t_id+1].start_char == landmark_span.start_char:
                    if e_t_id - window > 0:
                        previous_tokens = list(doc[e_t_id-window:e_t_id])
                    else:
                        previous_tokens = list(doc[0:e_t_id])

            for p_id, p_t in enumerate(previous_tokens):
                if p_t.text in spatial_indicator_list:
                    tmp_spa_list.append(p_t)
            landmark_list.append(tmp_spa_list)
            final_none_chunks.append(landmark_list)
        
        # to allocate different role
        landmark_dict = {}
        landmark_dict['landmarks'] = []
       
        for e_l_l, landmark_list in enumerate(final_none_chunks):
            each_landmark_dict = {}
            each_landmark_dict['id'] = e_l_l
            if not landmark_list[0][1]:      
                each_landmark_dict['text'] = landmark_list[0][0][-1].text
            else:
                    tmp_sub_landmark = each_landmark_dict 
                    for sub_id, each_sub_landmark in enumerate(landmark_list[:-1]):
                            tmp_sub_landmark['sub_landmark'] = {}
                            tmp_sub_landmark = tmp_sub_landmark['sub_landmark'] 
                            tmp_sub_landmark['text'] = each_sub_landmark[-1][-1].text
                            tmp_sub_landmark['spatial_indicator'] = each_sub_landmark[1]
   
                    each_landmark_dict['text'] = landmark_list[0][0][-1].text

            if landmark_list[-1]:
                    spatial_list = [spa_token.text for spa_token in landmark_list[-1]]
                    each_landmark_dict['spatial_indicator'] = ",".join(spatial_list)
            landmark_dict['landmarks'].append(each_landmark_dict)
        
        return landmark_dict

    


def get_spatial_indicator(test_sentence, landmark_list):
    landmark_text_list = list(list(zip(*landmark_list))[0])
    doc = nlp(test_sentence)
    window = 5
    spatial_phrase = []
    for token_id, each_token in enumerate(doc):
        if each_token.text in landmark_text_list:
            first_index = token_id - window if token_id - window > 0 else 0
            window_tokens = doc[first_index:token_id]
            for w_id, each_w_t in enumerate(window_tokens):
                if each_w_t.text in spatial_indicator_list:
                    spatial_phrase.append(" ".join([each_m.text for each_m in doc[first_index + w_id:token_id+1]]))

    return spatial_phrase




if __name__ == "__main__":
    
    '''
    # configuration operation
    path1 = 'tasks/R2R-pano/results/experiments_20200726-202958/_val_seen_epoch_135.json'
    path2 = 'tasks/R2R-pano/data/data/R2R_val_seen.json'
    path3 = 'tasks/R2R-pano/landmark_text_val_seen.json'
    #result_compare(path1, path2, path3)
    config_result  = get_configurations("Turn right, walk down hallway passed paintings,walk straight through double doors, turn left, go straight, turn left at the kitchen and stop by the refrigerator. ")
    print(config_result)
    # gold_path = 'tasks/R2R-pano/split_configuration_gold.txt'
    # comp_path = 'tasks/R2R-pano/split_configuration_train.txt'
    # compare_config(gold_path, comp_path)
    '''
    

    
    # motion indicatior
    '''
    all_example = open("/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/gold_data/split_configuration_gold.txt", "r").read()
    all_example = [each_example.strip().split('\n') for each_example in all_example.split('\n\n')]
    final_motion_indicator = []
    for each_example in all_example:
        temp_dict = defaultdict(list)
        for each_sentence in each_example[2:]:
            motion_indicator = get_motion_indicator2(each_sentence, motion_dict)
            temp_dict[each_example[0]].append((each_sentence, motion_indicator)) 
        final_motion_indicator.append(temp_dict)

    # with open("/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/component_data/motion_indicator.json",'w') as f_m_w:
        # json.dump(final_motion_indicator, f_m_w, indent=4)
    '''
    
    # landmark
    
    all_example = open("/VL/space/zhan1624/R2R-EnvDrop/r2r_src/components/config_split.txt", "r").read()
    all_example = [each_example.strip().split('\n') for each_example in all_example.split('\n\n')]
    final_landmarks = {}
    for each_example in all_example:
        for each_sentence in each_example[2:]:
            motion_indicator = get_motion_indicator2(each_sentence, motion_dict)
            landmark_list = get_landmark2(each_sentence, motion_dict)
            final_landmarks['motion_indicator'] = motion_indicator
            final_landmarks['landmarks'] = landmark_list['landmarks']
            final_landmarks['sentence'] = each_sentence

            break
        break
    with open("/VL/space/zhan1624/R2R-EnvDrop/r2r_src/components/landmark_comb.json",'w') as f_l_w:
        json.dump(final_landmarks, f_l_w, indent=4)
    print('quan')
    
   
   
    #motion_indicator dictionary
    
    # all_example = open("/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/gold_data/split_configuration_gold.txt", "r").read()
    # all_example = [each_example.strip().split('\n') for each_example in all_example.split('\n\n')]
    # final_motion_indicator = []
    # motion_indicator_list = []
    # for each_example in all_example:
    #     temp_dict = defaultdict(list)
    #     for each_sentence in each_example[2:]:
    #         motion_indicator_vocab = get_motion_dict(each_sentence)
    #         if motion_indicator_vocab not in motion_indicator_list:
    #             motion_indicator_list.append(motion_indicator_vocab)
    
            

    #spatial_indicator
    
    # all_example = open("/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/gold_data/split_configuration_gold.txt", "r").read()
    # all_example = [each_example.strip().split('\n') for each_example in all_example.split('\n\n')]
    # for each_example in all_example:
    #     temp_dict = defaultdict(list)
    #     for each_sentence in each_example[2:]:
    #         landmark_list = get_landmark(each_sentence, motion_dict)
    #         if len(landmark_list) == 6:
    #             get_spatial_indicator(each_sentence, landmark_list)

    #landmark_list = np.load('tasks/R2R-pano/data/data/component_data/landmarks/new_landmarks_feature/landmark_feature/landmark_train_feature.npy', allow_pickle=True).item()
   

    # generate configurations of trainning sets
    