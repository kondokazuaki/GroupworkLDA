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
def corpus_BoIW2gensim(corpus_BoIW) :
    
    corpus_gensim = [[(vid, corpus_BoIW[sid,vid]) for vid in range(corpus_BoIW.shape[1]) if corpus_BoIW[sid,vid] > 0] for sid in range(corpus_BoIW.shape[0])]
    return corpus_gensim


#--------------------------------------------------------------
def corpus_gensim2BoIW(corpus_gensim, num_IW_type) :

    num_scene = len(corpus_gensim)
    corpus_BoIW = np.zeros( (num_scene,num_IW_type) )

    for i in range(num_scene):
        for j,value in corpus_gensim[i]:
            corpus_BoIW[i,j] = value

    return corpus_BoIW


#--------------------------------------------------------------
# トピック間の最大cosを得る（＝トピック間の独立性の指標）
def get_max_cos_among_topics(model):
    TW_prob = get_TWprob(model)
    num_topics = model.num_topics

    max_cos = 0.0 
    for i in range(num_topics) :
        for j in range(i) :
            prob1 = TW_prob[:,i]
            prob2 = TW_prob[:,j]
            len1  = np.linalg.norm(prob1,ord=2)
            len2  = np.linalg.norm(prob2,ord=2)
            cos = float(np.dot(prob1,prob2)/(len1*len2))
            if cos > max_cos : max_cos = cos

    return max_cos
    

#--------------------------------------------------------------
#場面毎のトピック分布をnumpy.array型で得る．
def get_DTprob(model,corpus):
    DTprob = []
    pred = model.get_document_topics(corpus,minimum_probability=0)
    for a in pred:
        DTprob_i = [right for left,right in a]
        DTprob.append(DTprob_i)
    DTprob = np.array(DTprob)
    return DTprob


#--------------------------------------------------------------
#入力したモデルのトピック毎の単語出現確率をnumpy.array型で得る
def get_TWprob(model):
    TWprob = []
    num_topics = model.num_topics
    for i in range(num_topics):
        TWprob_i = model.get_topic_terms(topicid = i, topn = model.num_terms)
        TWprob_i = sorted(TWprob_i, key=lambda x: x[0])
        TWprob_i = [right for left,right in TWprob_i]
        TWprob.append(TWprob_i)
    TWprob = np.array(TWprob)
    TWprob = np.transpose(TWprob)
    return TWprob


#--------------------------------------------------------------
#perplexity（計算の都合で定義式を式変形して実装した結果，小数の出現量も扱える．もちろん単語出現量が全て整数の場合はperplexityの定義式通りの計算になる．）
def calc_perplexity(model, corpus_gensim):
    DTprob = get_DTprob(model,corpus_gensim)             #場面毎のトピック分布
    TWprob = get_TWprob(model)                           #トピック毎の単語生成確率
    corpus_BoIW = corpus_gensim2BoIW(corpus_gensim,model.num_terms)  #場面数☓語彙数で、各場面の各単語の出現量を表したもの

    #まず、各単語種の単語の出現率を求める(場面毎のトピック出現率☓各トピックにおける単語生成率を計算して和を取って、各場面のトピックの偏りを踏まえた各単語の出現率を得る）
    PofVocabs = DTprob @ np.transpose(TWprob)
    PofVocabs = PofVocabs + 10e-18 #エラー回避用に下駄を履かせる
    
    #モデルのもとで各文書が生成される確率のlogであるlog(p(w_d/M))の計算
    logp = corpus_BoIW * np.log(PofVocabs) #語彙が同じならばPofVocabsの値も同じ。同じ単語については、単語の出現量を値にかければ良い
    logp = np.sum(logp,-1) #全語彙分を足し合わせることで、モデルのもとで各文書が生成される確率のlogが求まる。
    Nd = np.sum(corpus_BoIW,-1) #各場面の単語の総出現量
    perplexity = np.exp( -np.sum(logp)/np.sum(Nd) )
    perplexities = [ np.exp(-logp[i]/Nd[i]) for i in range(len(Nd)) ]
    
    return perplexity, perplexities


#--------------------------------------------------------------
# モデル適合度（各場面においてトピック毎の単語生成分布とトピック分布を掛け算して，場面に対する単語生成分布を獲得し，実際の単語分布との類似度を計算する）
def calc_precision(model, corpus_gensim) :
    DTprob = get_DTprob(model,corpus_gensim)             #場面毎のトピック分布
    TWprob = get_TWprob(model)                           #トピック毎の単語生成確率
    PofVocabs = DTprob @ np.transpose(TWprob)            # 場面ごとの単語生成確率
    corpus_BoIW = corpus_gensim2BoIW(corpus_gensim,model.num_terms)  #場面数☓語彙数で、各場面の各単語の出現量を表したもの
    
    precisions = []
    for i in range(len(corpus_BoIW)) :
        prob = PofVocabs[i,:]
        hist = corpus_BoIW[i,:]
        lenp = np.linalg.norm(prob,ord=2)
        lenh = np.linalg.norm(hist,ord=2)
        cos  = float(np.dot(prob,hist)/(lenp*lenh))
        precisions.append(cos)

    return sum(precisions)/len(precisions), precisions

        

#--------------------------------------------------------------
# num_learn回学習し、モデル適合度が最も高いモデルを返す
# トピック間の独立性を考慮する（似たトピックを持つモデルをフィルタする）場合は，モデル適合度上位N%から最も独立性の高い試行を返す
def model_train(corpus_gensim, num_topic, vocabulary, num_learn=1, random_state_start=None, think_topic_independency=False):

    topN = 0.05
    
    # 
    id2word = dict()
    for i,IW in enumerate(vocabulary) :
        id2word[i] = IW

    # learning
    #models = [gensim.models.ldamodel.LdaModel(corpus=corpus_gensim,num_topics=num_topic,id2word=id2word) for i in range(num_learn)]
    models = []
    precisions = []
    for i in range(num_learn) :
        model = gensim.models.ldamodel.LdaModel(corpus=corpus_gensim,num_topics=num_topic,id2word=id2word)
        precision = calc_precision(model,corpus_gensim)[0]
        models.append(model)
        precisions.append(precision)
        #print(i)
        
    # select an optimal result w.r.t precision and topic independency  
    arglist = np.argsort(precisions)[::-1]
    argopt  = arglist[-1]    
    num_top = int(num_learn*topN)    
    if think_topic_independency and num_top>1:
        max_coss = [ get_max_cos_among_topics(models[i]) for i in arglist[0:num_top] ]
        argmin   = np.argsort(max_coss)[0]
        argopt   = arglist[argmin]
    opt_model = models[argopt]
        
    # calc some features
    opt_precision = calc_precision(opt_model,corpus_gensim)
    max_cos = get_max_cos_among_topics(opt_model)     
    #opt_perplexity = float(np.exp(-opt_model.log_perplexity(corpus_gensim)))
    opt_perplexity = calc_perplexity(opt_model, corpus_gensim)
    
    return opt_model, opt_precision, max_cos, opt_perplexity



#--------------------------------------------------------------
def draw_IW_occurance_prob(IW_prob, IW_label, prob_label, topic_label, max_prob, figsize, dpi, filename) :

    num_topic = IW_prob.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=num_topic, sharex=False, figsize=figsize, dpi=100)
    fig.subplots_adjust(left=0.2)
    for i in range(num_topic) :
        axes[i].barh(y=IW_label, width=IW_prob[:,i])
        axes[i].set_xticks(prob_label)
        axes[i].invert_yaxis()
        axes[i].set_xlabel(topic_label[i],size=32)
        axes[i].set_xlim(0,max_prob)
        axes[i].tick_params(labelsize=28)
        
        if i>0 : axes[i].tick_params(labelleft=False)

    plt.savefig(filename)


#--------------------------------------------------------------
def draw_topic_dist_indiv(topic_dist, scene_fit_values, time_label, topic_label, figsize, dpi, filename) :

    ts = 1
    time_label = np.array(time_label)
    num_topic = topic_dist.shape[1]
    fig, axes = plt.subplots(nrows=num_topic, ncols=1, sharex=False, figsize=figsize, dpi=100)
    for i in range(num_topic) :
        axes[i].plot(time_label, topic_dist[:,i], color='black')
        for j in range(0,len(time_label)-ts,ts) :
            alpha = sum(scene_fit_values[j:j+ts])/ts
            axes[i].fill_between(time_label, topic_dist[:,i], 0, facecolor='black', where=(time_label[j]<=time_label)&(time_label<=time_label[j+ts]), alpha=alpha )
        axes[i].set_ylabel(topic_label[i],size=32)
        axes[i].set_ylim(0,1)
        axes[i].tick_params(labelsize=28)
        if i<num_topic-1 : axes[i].tick_params(labelbottom=False)

    #axes[num_topic].plot(time_label, scene_fit_values, color='black')
    #axes[num_topic].set_ylabel('Scene precision', size=14)
    #axes[num_topic].set_ylim(0,1)
    #axes[num_topic].tick_params(labelsize=12)
    
    plt.savefig(filename)


#--------------------------------------------------------------
def draw_topic_dist_merge(topic_dist, time_label, topic_label, figsize, dpi, filename) :

    num_topic = topic_dist.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=figsize, dpi=100)
    dist = [0] * len(time_label)
    cm = plt.colormaps['tab20'].colors
    #cm = list(colors.TABLEAU_COLORS.values())

    for i in range(num_topic) :
        dist_prv = dist.copy()
        dist = dist + topic_dist[:,i]
        #axes.plot(time_label, list(dist), color=cm.colors[i])
        axes.fill_between(time_label, list(dist), list(dist_prv), facecolor=cm[i])

    axes.set_ylim(0,1)
    axes.tick_params(labelsize=12)

    plt.savefig(filename)
