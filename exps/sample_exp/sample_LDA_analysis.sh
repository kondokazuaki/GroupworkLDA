#! /bin/bash

iw_dir="../../tools/interaction_word_definition/ver_2023-02/"
mod_dir="../../tools:"$iw_dir


# 1. making Bag-of-Words from annotation data (but inconstant time duration)
target="group1"
duration="0:1080"
src_behavior_list=$iw_dir"/behavior_list.dat"
src_object_list=$iw_dir"/object_list.dat"
src_IW_vocabulary=$iw_dir"/iw_list.dat"
src_annotation_list="src_annotation_list.dat"
src_annot_dir="../groupwork_annotation"
cat $src_annot_dir/"annot_"$target.dat | awk -v srcdir=$src_annot_dir '{print $1,srcdir"/"$2}' > $src_annotation_list

dst_dir="BoIW_data"
if [ ! -d $dst_dir ] ; then
    mkdir $dst_dir
fi
dst_BoIW_path=$dst_dir"/BoIW_"$target".json"

#python3 ../../tools/ELAN_annotation_to_BoIW.py --src_behavior_list $src_behavior_list --src_object_list $src_object_list --src_IW_vocabulary $src_IW_vocabulary --src_annotation_list $src_annotation_list --duration $duration --dst_BoIW_path $dst_BoIW_path --module_dir $mod_dir --verbose --check_data


# 2. making corpus from BoW  with constant time duration and time step
echo $target,$dst_BoIW_path > target_BoIW_files.dat
dst_dir="corpus_data"
if [ ! -d $dst_dir ] ; then
    mkdir $dst_dir
fi
dst_corpus_path=$dst_dir"/corpus_"$target".json"

#python3 ../../tools/BoIW_to_corpus.py --scene_duration 10 --scene_step 1 --src_BoIW_file_list target_BoIW_files.dat --tfidf --idf_base 2 --idf_bias 0 --dst_corpus $dst_corpus_path --check_data --module_dir $mod_dir --verbose


# 3. apply LDA
cat $src_IW_vocabulary > iw_list_selected.dat
num_topics=`echo {3..5} | sed -e 's/ /,/g'`
dst_dir="LDA_result"
if [ ! -d $dst_dir ] ; then
    mkdir $dst_dir
fi
dst_LDA_path=$dst_dir"/LDA_result.json"

#python3 ../../tools/LDA_analysis.py --src_corpus $dst_corpus_path --list_num_topic $num_topics  --BoIW_normalize --dst_LDA_path $dst_LDA_path --check_data --module_dir $mod_dir --verbose --src_IW_vocabulary_selected iw_list_selected.dat


# 4. visualize the result
num_topic=5
dst_visualize_path=$dst_dir"/LDA_result_visualize_"$num_topic"topics"
#python3 ../../tools/LDA_result_visualize.py --src_LDA_result $dst_LDA_path --src_num_topic $num_topic --max_prob 0.3 --pick_IW 3 --dst_visualize_path $dst_visualize_path --check_data --module_dir $mod_dir --verbose
