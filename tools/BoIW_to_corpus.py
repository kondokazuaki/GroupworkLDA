import os
import argparse
import sys
import math
import numpy as np
import json
import datetime


#--------------------------------------------------------------
def arg_parse() :
    
    # Parse command line arguments    
    parser = argparse.ArgumentParser(description='Generate a group work corpus from Bag-of-Interaction-Word file(s) with concatnating and/or weightening')

    # input
    parser.add_argument('--src_BoIW_file_list', required=True,
                        help="A list of participant's behaviors for elements of interaction words. ")

    parser.add_argument('--scene_duration', required=True, type=float, default='10.0',
                        help="How long time in sec. is assumed as 1 scene. Default=10sec.")

    parser.add_argument('--scene_step', required=True, type=float, default='5.0',
                        help="With how long time step in sec. you want to generate scenes.Default=5sec.")

    parser.add_argument('--tfidf', action='store_true',
                        help='with weightening by TF-IDF')

    parser.add_argument('--idf_base', required=False, type=float, default='2.0',
                        help='base value b of logarithm in log_b(N/n)+a for IDF. default=2.0')

    parser.add_argument('--idf_bias', required=False, type=float, default='1.0',
                        help='bias value a in log_b(N/n)+a for IDF. default=1.0')
    
    
    parser.add_argument('--module_dir', required=False, default=False,
                        help="A directory in which additional module file(s) exist")

    
    # output
    parser.add_argument('--dst_corpus_path', required=True,
                        help="path to generated corpus file (metadata). Body data is saved into additional file(s)")

    parser.add_argument('--check_data', action='store_true',
                        help="save also intermediate result as check data")
                

    # others
    parser.add_argument('--verbose', action='store_true',
                        help='control verbose mode')

    return parser.parse_args()


#--------------------------------------------------------------
def calc_overlap(range1, range2) :

    time_start = max(range1[0],range2[0])
    time_end   = min(range1[1],range2[1])
    duration = time_end-time_start

    if duration<0 :
        print('[Warning] no overlap region between given ranges :', range1, range2)

    return duration


#--------------------------------------------------------------
def main() :

    args = arg_parse()

    if args.module_dir!=False :
        for path in args.module_dir.split(':') :
            sys.path.append(path)        
    import interaction_word as iw    
    
    path = args.dst_corpus_path.split('.')[0]
    if args.check_data :
        fc = open(path+"_checkdata.dat", mode='w')

        
    #
    scene_duration = args.scene_duration
    scene_step     = args.scene_step

    # load BoIW(s)
    BoIW_group_names = []
    BoIW_collection = []
    vocabulary = dict()
    num_IW_type = set()
    with open(args.src_BoIW_file_list) as f :        
        for line in f :
            words = line.strip().split(',')
            BoIW_group_names.append(words)            
            with open(words[1]) as fb :
                metadata = json.load(fb)
            vocabulary = metadata['Vocabulary']
            num_IW_type.add(len(vocabulary))
            BoIW_body_filename = os.path.join(metadata['Root_path'], metadata['BoIW_bodydata'])
            with open(BoIW_body_filename) as fb :                
                BoIW_group = []
                fb.__next__()
                for line in fb :
                    word = line.strip().split(',')
                    boiw = [int(val) for val in word[3:]]
                    BoIW_group.append( [word[0]] + [float(val) for val in word[1:3]] + boiw )
                BoIW_collection.append(BoIW_group)
    num_groups = len(BoIW_collection)
    if len(num_IW_type) >1 :
        print('[Error] vacabulary mismatch between multiple BoIWs')
        sys.exit()
    num_IW_type = num_IW_type.pop()

    
    ################
    # convert scenes with different durations into those with the same durations
    # original scenes :  |----------|---|------|-------|----|---
    # new scenes      :  |-----|-----|------|------|------|-----
    if args.verbose : print('[Report] converting scenes with different durations into those with the same durations...')
    if args.check_data : print('[converted scenes]',file=fc)
    corpus = []
    num_scenes = [0]
    for BoIW_group in BoIW_collection :
        time_start = BoIW_group[0][1]
        time_end   = BoIW_group[-1][2]
        num_scene = int((time_end-time_start-scene_duration)/scene_step)+1
        scene_time_start = [ float(i)*scene_step+time_start  for i in range(num_scene) ]

        start_idx = 0
        for i in range(num_scene) :
            new_BoIW = np.zeros(num_IW_type, dtype=float)
            tsst = scene_time_start[i]
            j = start_idx
            while tsst > BoIW_group[j][2] :
                j += 1
            start_idx = j
            check_duration = 0
            while j<len(BoIW_group) and tsst+scene_duration > BoIW_group[j][1] :
                overlap_duration = calc_overlap( [tsst,tsst+scene_duration], BoIW_group[j][1:3] )
                new_BoIW += np.array(BoIW_group[j][3:])*overlap_duration
                check_duration += overlap_duration
                j += 1                
            if args.check_data : print(new_BoIW, 'total', check_duration, file=fc)
            if check_duration==0 : continue
            if check_duration<scene_duration :
                new_BoIW = [ val*scene_duration/check_duration for val in new_BoIW ]            
            corpus.append(new_BoIW)

        num_scenes.append(len(corpus)) 

    num_total_scenes = len(corpus)
    corpus = np.array(corpus)
    
        
    ################        
    # apply weightening
    if args.tfidf :
        if args.verbose : print('[Report] applying TF-IDF...')
        # inverse document frequency (smooth)
        # - 0.0 for a word not appearing in any scenes
        # - 1.0 for a word appearing in all scenes
        # - w>1.0 for a word appearing in only portions of scenes 
        word_existence_count = np.count_nonzero(corpus>0, axis=0)
        if args.check_data : print('\n# of scenes in which a target word exists :\n',word_existence_count, file=fc)

        idfs = np.zeros(num_IW_type)
        for i in range(num_IW_type) :
            if word_existence_count[i]>0 : idfs[i] = math.log(float(num_total_scenes)/float(word_existence_count[i]),args.idf_base)+args.idf_bias
        if args.check_data : print('\nIDF values :\n',idfs,file=fc)    
        # apply idf values to original TF data 
        for i,idf in enumerate(idfs) :
            corpus[:,i] *= idf


    ##################
    # save results
    if args.verbose : print('[Report] saving results...')

    # meta data
    result = dict()
    result['Generated_date'] = datetime.datetime.now().isoformat()
    result['Root_path'] = os.getcwd()
    result['Scene_duration(sec)'] = scene_duration
    result['Scene_step(sec)'] = scene_step
    result['#_of_scenes'] = num_total_scenes
    scenes_each_group = []
    for i in range(num_groups) :
        scenes_each_group.append(BoIW_group_names[i] + [ num_scenes[i]+1, num_scenes[i+1] ])
    result['Scenes_each_group'] = scenes_each_group        
    result['Vocabulary'] = vocabulary
        
    if args.tfidf :
        result['IDF_base'] = args.idf_base
        result['IDF_bias'] = args.idf_bias
        result['IDF_val'] = list(idfs)

    result['Corpus_bodydata'] = path+"_bodydata.dat"
    if args.check_data : result['check_data'] = path+"_checkdata.dat"
        
    with open(path+'.json', mode='w', encoding='utf-8') as f :
        d = json.dumps(result, ensure_ascii=False, indent=2)
        f.write(d)

    
    # corpus body data
    np.savetxt(fname=path+'_bodydata.dat', X=corpus, fmt='%.2f', delimiter=',', header='GroupWork Activity Corpus = '+str(num_total_scenes)+' scenes x '+str(num_IW_type)+' IWs')

                

    fc.close()
                

if __name__ == "__main__":

    main()
