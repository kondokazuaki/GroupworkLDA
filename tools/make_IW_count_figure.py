import os
import argparse
import sys
import numpy as np
import datetime
import json
#import gensim
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#import pickle


#--------------------------------------------------------------
def arg_parse() :
    
    # Parse command line arguments    
    parser = argparse.ArgumentParser(description='Make a figure of simple IW counts along time (to be compared to LDA result)')

    # input
    parser.add_argument('--src_corpus', required=True,
                        help=".json corpus file (metadata) to be analyzed. This will be an output of 'BoIW_to_corpus.py' for a target group without TF-IDF and normalize")

    parser.add_argument('--target_counting_IW_list', required=True,
                        help="A list of counting target IWs selected from the vocabulary. See sample_count_IW_list.dat")

    
    #parser.add_argument('--module_dir', required=False, default=False,
    #                    help="A directory in which additional module file(s) exist")

    
    # output
    parser.add_argument('--dst_IW_count_file', required=False,
                        help="filename of output datafile")
        
    parser.add_argument('--dst_IW_count_img', required=True,
                        help="filename of output image")

    parser.add_argument('--check_data', action='store_true',
                        help="save also intermediate result as check data")
    
    
    # others
    parser.add_argument('--verbose', action='store_true',
                        help='control verbose mode')
    

    return parser.parse_args()



#--------------------------------------------------------------
def main() :

    args = arg_parse()
    
    #if args.module_dir!=False :
    #    for path in args.module_dir.split(':') :
    #        sys.path.append(path)        
    #import interaction_word as iw    
    #import LDA_utils as utl

    # load counting target IWs
    counting_IW_lists = []
    with open(args.target_counting_IW_list) as f :
        for line in f :
            words = line.strip().split(',')
            counting_IW_lists.append(words)
    num_target = len(counting_IW_lists)
    print(counting_IW_lists)
    
    
    # load corpus
    with open(args.src_corpus) as fb :
        corpus_metadata = json.load(fb)
    corpus_filename = os.path.join(corpus_metadata['Root_path'], corpus_metadata['Corpus_bodydata'])
    corpus = np.loadtxt(corpus_filename, delimiter=',')
    vocabulary = corpus_metadata['Vocabulary']
    scene_duration = corpus_metadata['Scene_duration(sec)']
    scene_step     = corpus_metadata['Scene_step(sec)']
    num_scenes     = corpus_metadata['#_of_scenes']
    time_series    = [ i*scene_step+0.5*scene_duration for i in range(num_scenes) ]

    
    # counting target IWs
    IW_counts = np.zeros( (num_scenes,num_target) )
    for i,IWs in enumerate(counting_IW_lists) :
        for IW in IWs :
            if IW in vocabulary :
                IW_counts[:,i] += corpus[:, list(vocabulary).index(IW)]/float(len(IWs))
            else :
                print('[Warning] given IW is not defined in the vocabulary')
        IW_counts[:,i] = IW_counts[:,i]/max(IW_counts[:,i])

        
    # save to file
    if args.dst_IW_count_file != None :
        with open(args.dst_IW_count_file, 'w') as f :
            for i in range(len(time_series)) :
                print('{:.5f}'.format(time_series[i]), end='', file=f)
                for j in range(num_target) :
                    print('\t', '{:.5f}'.format(IW_counts[i,j]), sep='', end='', file=f)
                print('\n', sep='', end='', file=f)
            
                
    # draw figure
    fig,ax = plt.subplots(figsize=[25.6,4.8], dpi=100)
    ax.set_xlabel('group work time', size=32)
    ax.set_ylabel('interaction\nword freq.', size=32)
    ax.tick_params(labelsize=28)
    ax.set_ylim(0,1.2)
    for i in range(num_target) :
        ax.plot(time_series, IW_counts[:,i], label=str(counting_IW_lists[i]), linewidth=3)
    plt.legend()
    plt.savefig(args.dst_IW_count_img)
    
    #fig,ax = plt.subplots(nrows=num_target, ncols=1, figsize=[25.6,4.8], dpi=100)
    #for i in range(num_target) :
    #    ax[i].set_xlabel('group work time')
    #    ax[i].set_ylabel('IW count\n(normalized)')
    #    ax[i].set_ylim(0,1.2)    
    #    ax[i].plot(time_series, IW_counts[:,i], label=str(counting_IW_lists[i]), linewidth=2)
    #plt.legend()

        

if __name__ == "__main__":

    main()
