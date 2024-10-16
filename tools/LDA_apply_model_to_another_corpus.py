import os
import argparse
import sys
import numpy as np
import datetime
import json
import gensim
import matplotlib.pyplot as plt
import pickle


#--------------------------------------------------------------
def arg_parse() :
    
    # Parse command line arguments    
    parser = argparse.ArgumentParser(description='Apply estimated LDA model to another BoIW corpus')
    
    # input
    parser.add_argument('--src_corpus', required=True,
                        help="Corpus file (metadata) to be analyzed ( generated by 'BoIW_to_corpus.py' ")
    
    parser.add_argument('--src_IW_vocabulary_selected', required=False,
                        help="A list of interaction words you want to analyze = portion of the original vocabulary")
    
    parser.add_argument('--src_LDA_model', required=False,
                        help="Metafile for the estimated LDA model files named with .json")

    parser.add_argument('--BoIW_normalize', action='store_true',
                        help='normalize total frequency in BoIW of each scene')
    
    parser.add_argument('--module_dir', required=False, default=False,
                        help="A directory in which additional module file(s) exist")
    
    # output
    parser.add_argument('--dst_LDA_path', required=True,
                        help="path for saving LDA analysis results. some additional words and/or extentions will be automatically added at its end to distinguish multiple results")
    
    
    # others
    parser.add_argument('--verbose', action='store_true',
                        help='control verbose mode')
    

    return parser.parse_args()



#--------------------------------------------------------------
def main() :

    args = arg_parse()
    
    if args.module_dir!=False :
        for path in args.module_dir.split(':') :
            sys.path.append(path)        
    import interaction_word as iw    
    import LDA_utils as utl
    
    path = args.dst_LDA_path.split('.')[0]
                                         
    # load corpus
    with open(args.src_corpus) as f :
        corpus_metadata = json.load(f)
    vocabulary = corpus_metadata['Vocabulary']
    corpus_body_filename = os.path.join(corpus_metadata['Root_path'], corpus_metadata['Corpus_bodydata'])
    corpus_org = np.loadtxt(corpus_body_filename, delimiter=',')
    num_scene = corpus_metadata['#_of_scenes']
    num_IW_type = len(vocabulary)
    
    
    ##################
    # corpus adjustment
    
    # load interaction word vocabulary (selected)
    vocabulary_slct = vocabulary
    if args.src_IW_vocabulary_selected :
        vocabulary_slct, vocabulary_slct_notation = iw.load_vocabulary(args.src_IW_vocabulary_selected)
        if args.verbose : print('Vocabulary(selected) : ', len(vocabulary_slct), ' words\n', vocabulary_slct)
        
    # extract selected BoIW
    corpus_slct = corpus_org
    if args.src_IW_vocabulary_selected :
        num_IW_type = len(vocabulary_slct)
        corpus_slct = np.zeros( (num_scene,num_IW_type) )
        for i,IW in enumerate(vocabulary_slct) :
            if IW not in vocabulary :
                print('[Error] selected IW does not exit in the vocabulary')
                sys.exit()
            corpus_slct[:,i] = corpus_org[ :,list(vocabulary.keys()).index(IW) ]
            

    # normalize BoIW
    corpus = corpus_slct.copy()
    if args.BoIW_normalize :
        num_normalize = 1000
        for i in range(num_scene) :
            sum_val = sum(corpus[i,:])
            if sum_val>0 : corpus[i,:] = corpus[i,:] *num_normalize/sum_val

    corpus_gensim = utl.corpus_BoIW2gensim(corpus)

                
    ##################
    # load estimated LDA model

    # load LDA model metadata
    with open(args.src_LDA_model) as f :
        LDA_metadata = json.load(f)
    src_path          = LDA_metadata['Root_path']
    num_topics        = LDA_metadata['Topic_nums']
    model_filenames   = LDA_metadata['Trained_models']
        
    # load models
    LDA_models = []
    for filename in model_filenames :                        
        with open(os.path.join(src_path,filename), mode='rb') as f:
            model = pickle.load(f)
            LDA_models.append(model)
            
            
    ##################
    # apply estimated LDA model to the given corpus    
    perplexities = []
    scene_perplexities = []
    model_precisions = []
    scene_precisions = []
    for model in LDA_models :
        perplexity = utl.calc_perplexity(model, corpus_gensim)
        perplexities.append(perplexity[0])
        scene_perplexities.append(perplexity[1])
        precision = utl.calc_precision(model,corpus_gensim)
        model_precisions.append(precision[0])        
        scene_precisions.append(precision[1])
        

    
    ##################
    # save results
    if args.verbose : print('[Report] saving results...')

    result = dict()
    result['Generated_date'] = datetime.datetime.now().isoformat()
    result['Root_path'] = LDA_metadata['Root_path']
    result['Applied_model_file'] = args.src_LDA_model 
    result['Topic_nums'] = num_topics
    result['Model_filenames'] = model_filenames
    result['Corpus_filename'] = corpus_body_filename
    result['Model_precisions']   = model_precisions
    result['Perplexities'] = perplexities
    result['Scene_fit'] = [ path+'_scene_fit_'+str(num)+'topics.dat' for num in num_topics ]
    with open(path+'.json', mode='w', encoding='utf-8') as f :
        d = json.dumps(result, ensure_ascii=False, indent=2)
        f.write(d)

    # analyzed corpus
    np.savetxt(fname=path+'_analyzed_BoIW.dat', X=corpus, fmt='%.2f', delimiter=',', header='GroupWork Activity Corpus = '+str(num_scene)+' scenes x '+str(num_IW_type)+' IWs')

        
    # scene fit
    for i in range(len(num_topics)) :
        data = []
        data.append(scene_precisions[i])
        data.append(scene_perplexities[i])
        data = (np.array(data)).transpose()
        np.savetxt(fname=result['Scene_fit'][i], X=data, fmt='%.2f', delimiter=',', header='Scene precison and perplexity along time')



    

if __name__ == "__main__":

    main()
