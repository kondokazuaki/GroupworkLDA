import sys
import re
import networkx as nx
import matplotlib as plt


#--------------------------------------------------------------
def load_object_list(filename=None) :
    object_list = dict()
    with open(filename) as f :    
        for line in f :
            words = line.strip().split()
            objs = words[1].split(sep=',') 
            for obj in objs :
                object_list[obj] = words[0]

    return object_list


#--------------------------------------------------------------
def load_behavior_list(filename=None) :
    behaviors = dict()
    with open(filename) as f :    
        for line in f :
            words = line.strip().split()
            behaviors[words[1]] = words[0]

    return behaviors


def behavior_check(behavior, person_list, behavior_list) :

    if len(behavior)<2 :
        print('[Error] insufficient elements in a given behavior')
        return False
    
    subject  = behavior[0]
    verb     = behavior[1]
        
    if subject not in person_list :
        print('[Error] a given subject does not exist in participants')
        return False
    if verb not in behavior_list.keys() :
        print('[Error] a given verb does not exist in a behavior list')
        return False

    return True
    

#--------------------------------------------------------------
def load_vocabulary(filename) :

    type2relation = dict()  # iw_type -> relation of the behaviors constructing the interaction word : {single, two-oneway, two-mutual, two-indirect}
    type2notation = dict()  # iw_type -> notation in Japanese
    
    with open(filename) as f :
        for line in f :
            words = line.strip().split()
            iw_type = words[0]
            iw_relation = words[1]
            iw_notation = words[2]
            #if iw_type in type2notation : # or iw_notation in type2notation.values() :
            if iw_notation in type2notation.values() :
                print('[Warning] given interaction word is overlapped to the registered one : ', iw_notation)                        
            type2relation[iw_type] = iw_relation
            type2notation[iw_type] = iw_notation
            
    return type2relation, type2notation
            

#--------------------------------------------------------------
class InteractionWord :
               
    def __init__(self) :        
        self.iw_type     = 'None'     # interaction word type defined in the vocabulary
        self.behavior1 = []           # [S,V,O] consisting of the behavior
        self.behavior2 = []           # [S,V,O] consisting of the behavior
        self.word_stc = nx.DiGraph()  # description of the interaction word as a directed graph

        
    def __eq__(self, other) :
        if not isinstance(other, InteractionWord) : return False
        return self.word_stc == other.word_stc

    def __hash__(self) :
        return hash(self.iw_type)

    
    def show(self, f=sys.stdout) :
        print(self.iw_type, self.behavior1, self.behavior2, file=f, sep=',')

    
    def draw(self) :
        nx.draw(self.word_stc, with_labels=True)


        
#--------------------------------------------------------------
def IW_from_single_behavior(behavior, vocabulary, person_list, behavior_list, object_list) :

    #--------------------------------------------------------------
    def generate_IW(IW_type, behavior) :        
        if len(IW_type)==1 :
            IW = InteractionWord()    
            IW.iw_type   = IW_type[0]
            IW.behavior1 = behavior
            for i in range(min(2,len(behavior)-1)) :
                IW.word_stc.add_edge(behavior[i],behavior[i+1])
            return IW        
        if len(IW_type)>1 :
            print('[Error] more than one IW match a given behavior')
        return False
    #--------------------------------------------------------------
        
    if not behavior_check(behavior, person_list, behavior_list) :
        return False
    
    verb = behavior_list[behavior[1]]
    
    # intransitive behavior
    if len(behavior)==2 :        
        IW_types = [ iwt for iwt, relation in vocabulary.items() if verb in iwt and 'object' not in iwt and 'person' not in iwt  ]
        return generate_IW(IW_types, behavior)

    
    # transitive behavior
    target = behavior[2]
    if target in person_list :
        # target==person
        IW_types = [ iwt for iwt, relation in vocabulary.items() if verb in iwt and 'person' in iwt and 'two-oneway' in relation  ]
        return generate_IW(IW_types, behavior)

    else :
        # target==object
        tgt = re.sub('（.+?）','',target) # （）で物体の属性を記載している部分を削除（本アノテーション方式では（）内に補足情報として他の物体が含まれていることがある）
        objs = [ obj for key, obj in object_list.items() if key in tgt ]  
        if not objs :
            print('[Error] no object matches the given target : ',tgt)
        if len(objs)>1 : print('[Error] more than 1 objects match to the given target ',target, objs)            
        IW_types = [ iwt for iwt, relation in vocabulary.items() if verb in iwt and objs[0] in iwt and 'single' in relation  ]
        return generate_IW(IW_types, behavior)

    return False


#--------------------------------------------------------------
def IW_from_two_behaviors(behavior1, behavior2, vocabulary, person_list, behavior_list, object_list) :

    #--------------------------------------------------------------
    def generate_IW(IW_type, behavior1, behavior2) :        
        if len(IW_type)==1 :
            IW = InteractionWord()    
            IW.iw_type   = IW_type[0]
            IW.behavior1 = behavior1
            IW.behavior2 = behavior2
            for i in range(min(2,len(behavior1)-1)) :
                IW.word_stc.add_edge(behavior1[i],behavior1[i+1])
            for i in range(min(2,len(behavior2)-1)) :
                IW.word_stc.add_edge(behavior2[i],behavior2[i+1])
            return IW        
        if len(IW_type)>1 :
            print('[Error] more than one IW match a given behavior')
        return False
    #--------------------------------------------------------------
    
    if not behavior_check(behavior1, person_list, behavior_list) :
        return False
    if not behavior_check(behavior2, person_list, behavior_list) :
        return False
    if len(behavior1)<3 or len(behavior2)<3 :
        print('[Error] both behavior must have own target')
        return False
    
    subject1  = behavior1[0]
    verb1     = behavior_list[behavior1[1]]
    target1   = behavior1[2]
    subject2  = behavior2[0]
    verb2     = behavior_list[behavior2[1]]
    target2   = behavior2[2]

    # same person
    if subject1==subject2 : return False    
    
    # mutual relation
    if subject1==target2 and subject2==target1 :    
        IW_types = [ iwt for iwt, relation in vocabulary.items() if (verb1+'+'+verb2 in iwt or verb2+'+'+verb1 in iwt) and 'two-mutual' in relation ]
        return generate_IW(IW_types, behavior1, behavior2)

    # indirect relation : person
    if target1==target2 and target1 in person_list :
        IW_types = [ iwt for iwt, relation in vocabulary.items() if (verb1+'+'+verb2 in iwt or verb2+'+'+verb1 in iwt) and 'person' in iwt and 'two-indirect' in relation ]
        return generate_IW(IW_types, behavior1, behavior2)
    
    # indirect relation : object
    if target1==target2 and target1 not in person_list :
        tgt = re.sub('（.+?）','',target1) # （）で物体の属性を記載している部分を削除（そこに他の物体が含まれていることがある）
        objs = [ obj for key, obj in object_list.items() if key in tgt ]
        if not objs : print('[Error] no object matches the given target')
        if len(objs)>1 : print('[Error] more than 1 objects match the given target ', target1, objs)
        IW_types = [ iwt for iwt, relation in vocabulary.items() if (verb1+'+'+verb2 in iwt or verb2+'+'+verb1 in iwt) and objs[0] in iwt and 'two-indirect' in relation ]
        return generate_IW(IW_types, behavior1, behavior2)
                
    return False
          


            
