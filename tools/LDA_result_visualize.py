import os
import argparse
import sys
import numpy as np
import datetime
import json
import gensim
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle


#--------------------------------------------------------------
def arg_parse() :
    
    # Parse command line arguments    
    parser = argparse.ArgumentParser(description='Visualize trained LDA model')

    # input    
    parser.add_argument('--src_LDA_result', required=True,
                        help="target trained LDA result you want to visualize. This is '.json' file generate from 'LDA_analysis.py'")    
    
    parser.add_argument('--src_num_topic', required=True, type=int,
                        help="which LDA model stored in '--src_LDA_result' you want to visualize, by indicating its # of topics")

    parser.add_argument('--max_prob', required=True, type=float,
                        help="max limit of IW occurance prob. in the figure")

    parser.add_argument('--pick_IW', required=False, type=int, default=0,
                        help="picking several IWs with highest prob. in each topic to show IW_occurance_prob., concretely")

    parser.add_argument('--module_dir', required=False, default=False,
                        help="A directory in which additional module file(s) exist")
    
    # output
    parser.add_argument('--dst_visualize_path', required=True,
                        help="path for saving visualized LDA results. some additional words and/or extentions will be automatically added at its end to distinguish multiple results")

    parser.add_argument('--check_data', action='store_true',
                        help="save also intermediate result as check data")
    
    
    # others
    parser.add_argument('--verbose', action='store_true',
                        help='control verbose mode')
    

    return parser.parse_args()




#--------------------------------------------------------------
def main() :

    args = arg_parse()

    # import additional func.
    if args.module_dir!=False :
        for path in args.module_dir.split(':') :
            sys.path.append(path)        
    import LDA_utils as utl
    
    # parse destination path
    dst_path = args.dst_visualize_path.split('.')[0]
    if args.check_data :
        fc = open(path+"_checkdata.dat", mode='w')
    
    # parse # of topics
    num_topic = args.src_num_topic
                                     
    # load LDA model metadata
    with open(args.src_LDA_result) as f :
        LDA_metadata = json.load(f)
    src_path          = LDA_metadata['Root_path']
    scene_duration    = LDA_metadata['Scene_duration(sec)']
    scene_step        = LDA_metadata['Scene_step(sec)']
    num_total_scenes  = LDA_metadata['#_of_scenes']
    scenes_each_group = LDA_metadata['Scenes_each_group'] 
    vocabulary        = LDA_metadata['Vocabulary']
    num_topics        = LDA_metadata['Topic_nums']
    model_filenames   = LDA_metadata['Trained_models']
    corpus_filename   = LDA_metadata['Analyzed_corpus']
    scene_fit_filenames = LDA_metadata['Scene_fit']
    
    
    # load visualization target model
    if num_topic not in num_topics :
        print('[Error] given num_topic was not used in the trained LDA models')
        sys.exit()
    model_filename = os.path.join(src_path,model_filenames[num_topics.index(num_topic)])
    with open(model_filename, mode='rb') as f:
        model = pickle.load(f)
    
    
    ##################
    # IW occurance prob. for each topic        
    IW_prob = utl.get_TWprob(model)
    IW_label   = list(vocabulary.keys())
    topic_label = [ "Topic_"+str(i+1) for i in range(num_topic) ]
    prob_label = [ val*args.max_prob/4.0 for val in range(5) ]
    figsize = np.array([25.6,19.2])
    img_path=dst_path+'_IW_prob.jpg'
    utl.draw_IW_occurance_prob(IW_prob, IW_label, prob_label, topic_label, args.max_prob, figsize, 100, img_path)
    
    # pick 3 IW with highest probs. up -> show 'OR' of those IWs
    if args.pick_IW>0 :
        picked_IW_indeces = set()
        for i in range(IW_prob.shape[1]) :
            prob_list = list(IW_prob[:,i])
            prob_sorted = sorted(prob_list, reverse=True)
            for j in range(args.pick_IW) :
                idx = prob_list.index(prob_sorted[j])
                picked_IW_indeces.add(idx)
        picked_IW_indeces = list(picked_IW_indeces)
        picked_IW_indeces.sort()

        IW_prob_picked = np.zeros((len(picked_IW_indeces),IW_prob.shape[1]))
        IW_label_picked = []
        for i, idx in enumerate(picked_IW_indeces) :
            IW_prob_picked[i,:] = IW_prob[idx,:]
            IW_label_picked.append(IW_label[idx])
        figsize = np.array([38.4,12.8])
        img_path=dst_path+'_IW_prob_picked.jpg'
        utl.draw_IW_occurance_prob(IW_prob_picked, IW_label_picked, prob_label, topic_label, args.max_prob, figsize, 100, img_path)

    
    ##################
    # topic dist.

    corpus_BoIW = np.loadtxt(os.path.join(src_path,corpus_filename), delimiter=',')
    corpus_gensim = utl.corpus_BoIW2gensim(corpus_BoIW)    
    topic_dist = utl.get_DTprob(model,corpus_gensim)

    # with scene fit
    scene_fit_filename = os.path.join(src_path, scene_fit_filenames[num_topics.index(num_topic)])
    scene_fit = np.loadtxt(scene_fit_filename, delimiter=',')[:,0]
    
    for i, group_data in enumerate(scenes_each_group) :
        group_name  = group_data[0]
        scene_start = group_data[2]
        scene_end   = group_data[3]
        num_scene   = scene_end - scene_start + 1
        
        time_label = [ float(i)*scene_step+scene_duration/2 for i in range(num_scene) ]

        figsize = np.array([38.4,9.6])
        img_path=dst_path+'_'+group_name+'_topic_dist_indiv.jpg'
        utl.draw_topic_dist_indiv(topic_dist[scene_start-1:scene_end,:], scene_fit[scene_start-1:scene_end], time_label, topic_label, figsize, 100, img_path)

        #figsize = np.array([25.6,2.4])
        #img_path=dst_path+'_'+group_name+'_topic_dist_merge.jpg'
        #utl.draw_topic_dist_merge(topic_dist[scene_start-1:scene_end,:], time_label, topic_label, figsize, 100, img_path)
    

    

if __name__ == "__main__":

    main()
