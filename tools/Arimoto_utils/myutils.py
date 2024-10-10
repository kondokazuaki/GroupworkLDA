import gensim
import scipy.io
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.gridspec as gs
import pickle

#gensimのコーパスの形から、matlabのCount（文書数☓語彙数の行列）の形へ
def corpus2Counts(corpus,numVocab=52):
    Counts = np.zeros((len(corpus),numVocab))
    for i in range(len(corpus)):
        the_corpus = corpus[i]
        for j,value in the_corpus:
            Counts[i,j] = int(value)
    return Counts

#gensimのCountの形から、matlabのコーパスの形へ
def Counts2corpus(Counts):
    return [[(vid, Counts[did,vid]) for vid in range(Counts.shape[1]) if Counts[did,vid] > 0] for did in range(Counts.shape[0])]

#matlabで作成したbagのCountsとVocabularyをgensimで用いられる形に変形する
def bag_matlab2gensim(Counts_path,Vocabulary_path):
    #matlabで作成したCountsを読み込む
    mat_Counts = scipy.io.loadmat(Counts_path) 
    if 'rawCounts_Basic' in Counts_path:
        counts = scipy.sparse.csc_matrix(mat_Counts['rawCounts_Basic'],dtype=np.int16)
    elif 'Basic' in Counts_path:
        counts = scipy.sparse.csc_matrix(mat_Counts['Counts_Basic'],dtype=np.int16)
    elif 'rawCounts' in Counts_path:
        counts = scipy.sparse.csc_matrix(mat_Counts['rawCounts'],dtype=np.int16)
    else:
        counts = scipy.sparse.csc_matrix(mat_Counts['Counts'],dtype=np.int16)
    #matlabで作成したVocabularyを、文字列のリスト型として読み込む
    mat_Vocabulary = scipy.io.loadmat(Vocabulary_path) 
    Vocabulary = mat_Vocabulary['Vocabulary'][0]
    Vocabulary = Vocabulary.replace('"','')
    Vocabulary = Vocabulary.replace("[","")
    Vocabulary = Vocabulary.replace("]","")
    Vocabulary = Vocabulary.split()
    #matlabのCountsをgensimのcorpusの形に変形する
    corpus = Counts2corpus(counts)
    #matlabのVocabularyをgensimのDictionaryの形に変形する
    dictionary = gensim.corpora.Dictionary([Vocabulary])
    return counts, Vocabulary, corpus, dictionary #順に、matlabの文書別単語出現頻度、matlabの語彙リスト、gensimのcorpus（文書別単語出現頻度に対応）、gensim用の辞書（語彙リストに対応）

#コーパスから用いる語彙idを選択肢、それらの単語のみ用いる（出現量を0にするだけ）
def use_words(corpus,vids):
    new_corpus = []
    for i in range(len(corpus)):
        new_corpus.append([(a,b) for a,b in corpus[i] if (a in vids)])
    return new_corpus

#コーパスから指定した語彙idリスト内にある単語のデータを除く（出現量を0にするだけ）
def remove_words(corpus,vids):
    new_corpus = []
    for i in range(len(corpus)):
        new_corpus.append([(a,b) for a,b in corpus[i] if not (a in vids)])
    return new_corpus


#語彙自体を変更するための関数。use_wordsの後にこの関数に通す形で組み合わせて使うことになる。
def use_words_real(corpus,Vocabulary,vids):
    new_Vocab = [Vocabulary[i] for i in vids]
    new_dic = gensim.corpora.Dictionary([new_Vocab])
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            the_tuple = corpus[i][j]
            for k in range(len(vids)):
                check_id = vids[k]
                if the_tuple[0] == check_id:
                    the_list = list(the_tuple)
                    the_list[0] = k
                    corpus[i][j] = tuple(the_list)
    return corpus,new_Vocab,new_dic


# 2022.11.27 by Kondo 
# コーパスフィルタリング：特定の語彙だけを取り出したコーパスを生成
def corpus_filtering_by_words(corpus_org,vocab_org,word_id_list):
    new_vocab = [vocab_org[i] for i in word_id_list]
    new_dic = gensim.corpora.Dictionary([new_vocab])
    new_corpus = []
    for i in range(len(corpus_org)):
        scene = []
        for j in range(len(corpus_org[i])):
            the_tuple = corpus_org[i][j]
            for k in range(len(word_id_list)):
                if the_tuple[0] == word_id_list[k]:
                    scene.append((k,the_tuple[1]))
        new_corpus.append(scene)
        
    return new_corpus,new_vocab,new_dic


# 2022.11.27 by Kondo
# それぞれのシーンに含まれている単語数をシーン間で正規化する（シーン内単語数で除して１に正規化）
def scene_normalization(corpus) :
    new_corpus = []
    for i in range(len(corpus)):
        num_words = 0
        for j in range(len(corpus[i])):
            num_words = num_words + corpus[i][j][1]
        scene = []
        for j in range(len(corpus[i])):
            word = corpus[i][j]
            scene.append((word[0],word[1]/num_words))
        new_corpus.append(scene)
    
    return new_corpus

        

#モデルのセーブのために作ったが、モデルに関わらず大抵のものはこれでセーブできる。
def save_model(model,path):
    with open(path, mode='wb') as f:
        pickle.dump(model, f)

#モデルのロード
def load_model(path):
    with open(path, mode='rb') as f:
        model = pickle.load(f)
    return model

#場面毎のトピック分布をnumpy.array型で得る。dummy=trueの場合は、場面の中心秒とデータのtが合うように、ダミーデータをデータ手前に加える（長さ10秒スライド1秒を想定して0~4番目にダミー）
def get_DTprob(model,corpus,dummy=True):
    DTprob = []
    pred = model.get_document_topics(corpus,minimum_probability=0)
    for a in pred:
        DTprob_i = [right for left,right in a]
        DTprob.append(DTprob_i)
    DTprob = np.array(DTprob)
    if dummy == True:
        #場面の中心のおよその秒数が場面番号になるようにする。つまり、場面長10秒、スライド時間1秒の場合、0番目の場面が5番目の場面になるように、ダミーを手前に加える。
        before_start = np.zeros((5,DTprob.shape[1]))
        DTprob = np.concatenate((before_start,DTprob),0) 
    return DTprob

#トピック毎の単語分布をnumpy.array型で得る
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

#場面毎のトピック分布をプロット。
def plot_DTprob(model,corpus):
    textsize = 18
    num_topics = model.num_topics
    DTprob = get_DTprob(model,corpus)
    fig, ax = plt.subplots(num_topics,1,figsize=(10,1.3*(num_topics)),sharex=True)
    plt.subplots_adjust(hspace=0.5)
    for i in range(num_topics):
        ax[i].xaxis.set_major_locator(MaxNLocator(10))
        ax[i].set_ylim(0,1)
        ax[i].tick_params(axis="y",labelsize=textsize)
        ax[i].bar(np.arange(len(DTprob[:,i])),DTprob[:,i],width=1)
        ax[i].set_ylabel("topic"+str(i),fontsize=textsize)
    plt.tick_params(labelsize=textsize)
    plt.show()

#指定したcorpusの指定した場面番号の単語の出現頻度を、生の値と、和が１になるように計算したものの２つの形で得る。
def get_DWfrequency(corpus,id,Vocabulary,dummy=True):
    #データ番号＝場面の中心秒となるように合わせた想定でidを指定した場合、指定したid-5番目のcorpusが対応するデータとなる。
    if dummy == True:
        id = id - 5
    if id < 0:
        return 0
    the_corpus = corpus[id]
    the_count = np.zeros(len(Vocabulary))
    for vid,num in the_corpus:
        the_count[vid] = num
    the_count_rate = the_count / sum(the_count)
    return the_count, the_count_rate

#指定したcorpusの指定した場面番号の単語の出現頻度をプロット
def plot_DWfrequency(corpus,id,Vocabulary):
    _,count_rate = get_DWfrequency(corpus,id,Vocabulary)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0,1)
    ax.bar(np.arange(len(Vocabulary)),count_rate)
    plt.show()

#複数のモデルのヒートマップを同一カラーバーで描画,adjust_leftは単語分布の間隔調整用
def heatmap_TWprobs(TWprob_list,Vocabulary,adjust_left = 0.4,vmax=None,xlabellist=True,figwidth=6.4):
    vmin = 0
    max_list = []
    for i in range(len(TWprob_list)):
        max_list.append(np.max(TWprob_list[i]))
    if vmax == None:
        vmax = max(max_list)
    fig, ax = plt.subplots(1,len(TWprob_list),sharey=True)
    fig.set_figwidth(figwidth)
    fig.subplots_adjust(left=adjust_left)
    for i in range(len(TWprob_list)):
        sns.set()
        TWprob = TWprob_list[i]
        if i == len(TWprob_list)-1:
            if i > 0:
                sns.heatmap(TWprob,yticklabels=False, xticklabels=xlabellist, square=True, linewidths=0.5, ax=ax[i],vmin=vmin,vmax=vmax)
            else:
                sns.heatmap(TWprob,yticklabels=False, xticklabels=xlabellist, square=True, linewidths=0.5, ax=ax,vmin=vmin,vmax=vmax)
        else:
            sns.heatmap(TWprob,cbar=False, yticklabels=False, xticklabels=xlabellist, square=True, linewidths=0.5, ax=ax[i],vmin=vmin,vmax=vmax)
        if type(ax) == list or type(ax) == np.ndarray:
            ax[i].set_yticks(np.arange(0.5,len(Vocabulary)+0.5),Vocabulary, fontname = "Noto Sans CJK JP",rotation = 0)
        else:
            ax.set_yticks(np.arange(0.5,len(Vocabulary)+0.5),Vocabulary, fontname = "Noto Sans CJK JP",rotation = 0)
    plt.show()


#トピック毎の単語生成確率のヒートマップを、モデルリスト内の各モデルに対し表示。
def plot_TWprob_list(model_list,Vocabulary,adjust_left=0.4,vmax=None,useids=None):
    #各モデルの単語生成確率を計算
    TWprob_list = []
    max_list = []
    num_model = len(model_list)
    if useids != None:
        Vocabulary = [Vocabulary[i] for i in useids]
    for i in range(num_model):
        model = model_list[i]
        TWprob = get_TWprob(model)
        if useids != None:
            TWprob = TWprob[useids]
        TWprob_list.append(TWprob)
        max_list.append(np.max(TWprob))
    #描画
    heatmap_TWprobs(TWprob_list,Vocabulary,adjust_left=adjust_left,vmax=vmax)

#定義通りに自作したperplexity（基本的にこちらを用いる）
def get_perplexity(model,corpus):
    DTprob = get_DTprob(model,corpus,dummy=False) #場面毎のトピック分布
    TWprob = get_TWprob(model) #トピック毎の単語生成確率
    Counts = corpus2Counts(corpus,model.num_terms) #場面数☓語彙数で、各場面の各単語の出現量を表したもの
    #まず、各単語種の単語の出現率を求める(場面毎のトピック出現率☓各トピックにおける単語生成率を計算して和を取って、各場面のトピックの偏りを踏まえた各単語の出現率を得る）
    PofVocabs = DTprob @ np.transpose(TWprob)
    #モデルのもとで各文書が生成される確率のlogであるlog(p(w_d/M))の計算
    logp = Counts * np.log(PofVocabs) #語彙が同じならばPofVocabsの値も同じ。同じ単語については、単語の出現量を値にかければ良い
    logp = np.sum(logp,-1) #全語彙分を足し合わせることで、モデルのもとで各文書が生成される確率のlogが求まる。
    Nd = np.sum(Counts,-1) #各場面の単語の総出現量

    perplexity = np.exp(- np.sum(logp) / (np.sum(Nd)+0.000000001) )
    return perplexity
        
    #kondo
    # 場面（時刻）ごとのperplexity
    #perplexity_t = np.exp(-logp/Nd) # '/' を要素ごとの割り算とみなして使っている
    # データ全体に対するperplexity（場面ごとのperplexityの算術平均）
    #perplexity_all = np.sum(perplexity_t)/len(Nd)
    


#コーパス内の各文書のperplexityを計算してplot。返り値としてperplexity_listを返すこともできる。
def plot_DTprob_sceneperplexity(model,corpus,show_perplexity_max=52):
    textsize = 18
    num_topics = model.num_topics
    DTprob = get_DTprob(model,corpus)
    fig, ax = plt.subplots(num_topics+1,1,figsize=(10,1.1*(num_topics+1)),sharex=True)
    plt.subplots_adjust(hspace=0.5)
    for i in range(num_topics):
        ax[i].xaxis.set_major_locator(MaxNLocator(10))
        ax[i].set_ylim(0,1)
        ax[i].tick_params(axis="y",labelsize=textsize)
        ax[i].bar(np.arange(len(DTprob[:,i])),DTprob[:,i],width=1)
        ax[i].set_ylabel("topic"+str(i),fontsize=textsize)
    perplexity_list = [get_perplexity(model,[doc]) for doc in corpus]
    ax[-1].set_ylim(0,show_perplexity_max) #表示値の上限
    ax[-1].xaxis.set_major_locator(MaxNLocator(10))
    ax[-1].set_ylabel("perp",fontsize=textsize)
    correc = 5 #correcは場面の中心時刻とx軸の番号を合わせるための補正数値
    ax[-1].plot(range(correc,len(corpus)+correc),perplexity_list)
    plt.tick_params(labelsize=textsize)
    plt.show()

#（データ番号、指定データ番号の確率分布）のnumpy行列で表現された２つの確率分布集合prob1とprob2を受け取り、prob1の各確率分布に対するprob2の各確率分布のKLdivergenceを計算。
def calc_KLdiv(prob1,prob2):
    KL = np.zeros((prob1.shape[0],prob2.shape[0]))
    for i in range(prob1.shape[0]):
        for j in range(prob2.shape[0]):
            KL[i,j] = gensim.matutils.kullback_leibler(prob1[i],prob2[j])
    return(KL)

#model1の各トピックの単語生成確率をp側、model2の各トピックの単語生成確率をq側とした場合のKLdivergenceを計算、表の形でプロット
def plot_TWprob_KLdiv(model1,model2):
    TWprob1 = get_TWprob(model1)
    TWprob2 = get_TWprob(model2)
    KL = calc_KLdiv(np.transpose(TWprob1),np.transpose(TWprob2))
    sns.set()
    ax = sns.heatmap(KL,square=True,annot=True,fmt='.4g',linewidths=0.5,vmin=0,vmax=6)
    ax.set_xlabel("q側のモデルの各トピック",fontname = "Noto Sans CJK JP")
    ax.set_ylabel("p側のモデルの各トピック",fontname = "Noto Sans CJK JP")
    plt.show()
    
