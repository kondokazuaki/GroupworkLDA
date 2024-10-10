import os
import argparse
import sys
import numpy as np
import datetime
import json


#--------------------------------------------------------------
# ELAN annotation
# tuple:(start_time, end_time, subject, behavior, list:[target1,target2,...])
def load_ELAN_annotation(filename=None, duration=[0.0,100.0], subject='who', behavior_list=dict() ) :
    annots = []
    with open(filename) as f :
        for line in f :
            words = line.strip().split()            
            time_start = max(duration[0], float(words[1]))
            time_end   = min(duration[1], float(words[2]))
            if time_start>=time_end:
                #print(words)
                continue
            targets = []
            if len(words)>3 :
                targets = words[3].split(sep='&')
            behaviors = [ behavior for behavior in behavior_list if behavior in words[0] ]
            if len(behaviors) == 0 : continue
            if len(behaviors)>1 :
                print('[Warning] more than 1 behaviors match to the given annotation')
                continue
            annots.append( (time_start,time_end,subject,behaviors[0],targets) )

    return annots


#--------------------------------------------------------------
def arg_parse() :
    
    # Parse command line arguments    
    parser = argparse.ArgumentParser(description='Generate Bag-of-Interaction-Word from groupwork annotation with ELAN')

    # input
    parser.add_argument('--src_behavior_list', required=True,
                        help="A list of participant's behaviors for elements of interaction words. ")

    parser.add_argument('--src_object_list', required=True,
                        help="A list of objects for elements of interaction words.")
    
    parser.add_argument('--src_IW_vocabulary', required=True,
                        help="A list of interaction words defined by relation of the behavbiors. ")
    
    parser.add_argument('--src_annotation_list', required=True,
                        help="Annotation lists of each participant's behaviors in the target groupwork")

    parser.add_argument('--duration', required=True,
                        help="Time period you want to analyze in the target groupwork, as 'start:end' form in second.")

    parser.add_argument('--module_dir', required=False, default=False,
                        help="A directory in which additional module file(s) exist")
    
    # output
    parser.add_argument('--dst_BoIW_path', required=True,
                        help="path to generated Bag-of-Interaction Word file (metadata). Body data is saved into additional file(s)")

    parser.add_argument('--check_data', action='store_true',
                        help="save also intermediate result as check data")

    # others
    parser.add_argument('--verbose', action='store_true',
                        help='control verbose mode')

    return parser.parse_args()


def main() :

    args = arg_parse()
    
    if args.module_dir!=False :
        for path in args.module_dir.split(':') :
            sys.path.append(path)        
    import interaction_word as iw    

    path = args.dst_BoIW_path.split('.')[0]
    if args.check_data :
        fc = open(path+"_checkdata.dat", mode='w')

        
    # parse target duration
    duration = [ float(t) for t in args.duration.split(sep=':') ] 
    if args.verbose : print('Duration : ', duration)
        
    # load participant's behavior list
    behavior_list = iw.load_behavior_list(args.src_behavior_list)
    if args.verbose : print('Behavior list : ', len(behavior_list), '\n', behavior_list)
    
    # load object list
    object_list = iw.load_object_list(args.src_object_list)
    if args.verbose : print('Object list : ', len(object_list), '\n', object_list)

    # load interaction word vocabulary
    vocabulary, vocabulary_notation = iw.load_vocabulary(args.src_IW_vocabulary)
    if args.verbose : print('Vocabulary : ', len(vocabulary), ' words\n', vocabulary)

    
    # -------------------------------------------------------------------------
    # 0. load ELAN annotations
    if args.verbose : print('[Report] loading annotations...')
    annot_files = []
    with open(args.src_annotation_list) as f :
        for line in f :
            words = line.strip().split()
            annot_files.append(words[0:2])    
    annots = []
    person_list = []
    for annot_file in annot_files :
        person_list.append(annot_file[0])
        annots += load_ELAN_annotation(annot_file[1],duration,annot_file[0],behavior_list)

    if args.verbose :
        print('Annotations : ',len(annots), ' annots')
        #for annot in annots : print(annot)

        
    # -------------------------------------------------------------------------
    # 1. convert [ behaviors with overlapped relation in timeline] to [independent scenes]
    # scene
    # tuple : (start_time, end_time, sorted_list:[annot_idx1,idx2...])
    if args.verbose : print('[Report] converting annotations if time overppled ...')
    time_series = []
    for i, annot in enumerate(annots) :
        time_series.append([float(annot[0]), i, 'start'])
        time_series.append([float(annot[1]), i, 'end'])
    time_series.sort(key=lambda x:x[0])
    
    annot_idxs = set()
    previous_time = 0
    scene_series = []
    for ts in time_series :
        time = ts[0]
        idx  = ts[1]
        se_flag = ts[2]
        # register scene
        if previous_time<time : scene_series.append((previous_time,time,sorted(annot_idxs)))
        # update ongoing scene
        previous_time = time                                    
        if se_flag=='start' :
            annot_idxs.add(idx)
        if se_flag=='end' :
            annot_idxs.remove(idx)
                    
    if args.verbose :
        print('Scenes : ',len(scene_series), ' scenes')
        #for scene in scene_series : print(scene)
            
            
    # -------------------------------------------------------------------------
    # 2. generate interaction word collection
    if args.verbose : print('[Report] generating interaction words...')
    time_segments = []
    IW_collection = []
    for scene in scene_series :
        # put duration to 'time_segments'
        time_segments.append( (scene[0],scene[1]) )

        IW_group = set()
        # IW with single behavior
        for annot_idx in scene[2] :
            annot = annots[annot_idx]
            subject  = annot[2]
            verb     = annot[3]
            targets  = annot[4]

            if not targets :
                IW = iw.IW_from_single_behavior([subject,verb], vocabulary, person_list, behavior_list, object_list)
                if IW!=False : IW_group.add(IW)
            else :
                for target in targets :
                    IW = iw.IW_from_single_behavior([subject,verb,target], vocabulary, person_list, behavior_list, object_list)
                    if IW!=False : IW_group.add(IW)

        # IW with two behaviors
        for annot_idx1 in scene[2] :
            annot1 = annots[annot_idx1]
            subject1 = annot1[2]
            verb1    = annot1[3]
            targets1 = annot1[4]
            if not targets1 :
                continue
            for annot_idx2 in scene[2] :
                if annot_idx1 >= annot_idx2 :
                    continue
                annot2 = annots[annot_idx2]
                subject2 = annot2[2]
                verb2    = annot2[3]
                targets2 = annot2[4]
                if not targets2 :
                    continue

                for target1 in targets1 :
                    for target2 in targets2 :
                        IW = iw.IW_from_two_behaviors([subject1,verb1,target1], [subject2,verb2,target2], vocabulary, person_list, behavior_list, object_list)
                        if IW!=False : IW_group.add(IW)

        # add IW_group being sorted to IW_collection 
        IWs = []
        for key in vocabulary.keys() :
            for IW in IW_group :
                if IW.iw_type==key : IWs.append(IW)        
        IW_collection.append(IWs)        

    if args.check_data:
        print('[Scene structure]\n', file=fc)
        for i,time_segment in enumerate(time_segments) :
            print('scene_'+str(i), time_segment[0], time_segment[1], len(IW_collection[i]), file=fc, sep=',') 
            for IW in IW_collection[i] : IW.show(fc)
            print('\n',file=fc)
                        
            
    # -------------------------------------------------------------------------
    # 3. generate BoIW from IW collection = conuting # of IW for each IW_type (not distinguishing subject and targets in IW graph)
    if args.verbose : print('[Report] counting interaction words...')
    BoIW = np.zeros( (len(vocabulary),len(time_segments)) )
    for t, IW_group in enumerate(IW_collection) :                
        for idx, IW_type in enumerate(vocabulary.keys()) :
            hit = [ 1 for IW in IW_group if IW.iw_type==IW_type ]
            BoIW[idx][t] += len(hit)

            
        
    ##################
    # save results
    if args.verbose : print('[Report] saving results...')
    
    # meta data
    result = dict()
    result['Generated_date'] = datetime.datetime.now().isoformat()
    result['Root_path'] = os.getcwd()
    result['Input_annotations'] = [ line[1] for line in annot_files ]
    result['Participants'] = person_list
    result['Object_list'] = object_list
    result['Behavior_list'] = behavior_list
    result['#_of_IW_defined_in_vocabulary'] = len(vocabulary)
    result['Vocabulary'] = vocabulary
    result['#_of_scenes'] = len(time_segments)
    result['BoIW_bodydata'] = path+"_bodydata.dat"
    if args.check_data : result['check_data'] = path+"_checkdata.dat"
    
    with open(path+'.json', mode='w', encoding='utf-8') as f :
        d = json.dumps(result, ensure_ascii=False, indent=2)
        f.write(d)
    
    # BoIW bodydata
    with open(path+"_bodydata.dat", mode='w') as f :
        print('Index,TimeStart,TimeEnd,BoIWs',file=f)
        for i,boiw in enumerate(BoIW.T) :
            print('scene_'+str(i),time_segments[i][0], time_segments[i][1], file=f, sep=',', end='')
            print((',{:.0f}'*len(boiw)).format(*boiw),file=f,sep=',')



    if args.check_data : fc.close()
            

if __name__ == "__main__":

    main()
