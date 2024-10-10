#! /bin/bash


iw_dir="./tools/interaction_word_definition/ver_2023-02/"
mod_dir="./tools:"$iw_dir

targets="group1 group2"

#-------------------    
BoIW_dir="BoIW_data"
count_dir="IW_count"
if [ ! -d $count_dir ] ; then
    mkdir $count_dir
fi

for target in $targets ; do

    src_BoIW_path=$BoIW_dir"/BoIW_"$target".json"
    echo $target,$src_BoIW_path > target_BoIW_files.dat
    dst_corpus_path=$count_dir"/corpus_raw_"$target".json"

    #python3 ../../tools/BoIW_to_corpus.py --scene_duration 10 --scene_step 5 --src_BoIW_file_list target_BoIW_files.dat --dst_corpus $dst_corpus_path --check_data --module_dir $mod_dir --verbose

done


#-------------------

src_IW_count_list="sample_IW_count_list.dat"

for target in $targets ; do

    src_corpus_path=$count_dir"/corpus_raw_"$target".json"
    dst_img_path=$count_dir"/IW_count_"$target".jpg"
    
    python3 ../../tools/make_IW_count_figure.py --src_corpus $src_corpus_path --target_counting_IW_list $src_IW_count_list --dst_IW_count_img $dst_img_path --check_data --verbose
done


