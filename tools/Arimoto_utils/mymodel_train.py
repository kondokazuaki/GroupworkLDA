from cmath import nan
import gensim
import scipy.io
import scipy.sparse
import numpy as np
import myutils

#num_learn回学習し、perplexityが最も低いモデルと、学習した全モデルのリストと、対応するperplexityのリストを返す
def model_train(corpus,num_topics,id2word,random_state_start=None,num_learn=100):
    if random_state_start == None: #random_stateを指定せずに学習
        models = [gensim.models.ldamodel.LdaModel(corpus=corpus,num_topics=num_topics,id2word=id2word) for i in range(num_learn)]
    else: #random_stateを指定して学習。
        models = [gensim.models.ldamodel.LdaModel(corpus=corpus,num_topics=num_topics,id2word=id2word,random_state=random_state_start+i) for i in range(num_learn)]
    perplexitys = [myutils.get_perplexity(models[i],corpus) for i in range(num_learn)]
    arglist = np.argsort(perplexitys)
    argmin = arglist[0]
    min_model = models[argmin]
    return min_model, models, perplexitys


# 2022.11.27 by Kondo
def model_train_save(corpus,num_topics,id2word,random_state_start=None,num_learn=100,filename_model_opt, filename_model_all, filename_perp_opt, filename_perp_all):

    # model learning
    print('now learning ...')
    model_opt, model_all, perp = model_train(corpus, num_topics, id2word, random_state_start, num_learn)    
    print('done')

    # calc perplexity
    print('calclating perplexity ...')
    perp_opt = [ myutils.get_perplexity(model_opt,corpus[j]) for j in range(len(corpus)) ]
    perp_all = []
    for i in range(num_learn) :
        ps = [ myutils.get_perplexity(model_all[i],corpus[j]) for j in range(len(corpus)) ]
        perp_all.append(ps)
    print('done')

    # save
    print('saving learned models and perplexities ...')
    myutils.save_model(model_opt, filename_opt)
    myutils.save_model(model_all, filename_all)
    with open(filename_perp_opt, 'w') as f :
        csv.writer(f).writerow(perp_opt)
    with open(filename_perp_all, 'w') as f :
        csv.writer(f).writerows(perp_all)
    print('done')
