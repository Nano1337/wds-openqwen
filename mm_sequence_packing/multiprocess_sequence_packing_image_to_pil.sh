#!/bin/bash
start_id=$1
num_proc=$2 # on our server we use 4; you can enlarge this based on you cpu cores
tars_per_gpu=$3 # the number of tar files processed by each process
dataset=$4
data_root=$5

start=`date +"%s"`

i=0
while [ $i -lt $num_proc ]; do
  {
    echo $i
    if [ "$dataset" == "obelics" ]; then
      mkdir ${data_root}/obelics/obelics_single_pkl_pil
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path ${data_root}/obelics_webdataset --save-path ${data_root}/obelics_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$dataset" == "synthdog" ]; then
      mkdir ${data_root}/synthdog-en/synthdog_single_pkl_pil 
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path ${data_root}/synthdog_webdataset --save-path ${data_root}/synthdog_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$dataset" == "ccs" ]; then
      mkdir ${data_root}/ccs/ccb_single_pkl_pil
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path ${data_root}/ccs_webdataset --save-path ${data_root}/ccs_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$dataset" == "laion" ]; then
      mkdir ${data_root}/laion/laion_single_pkl_pil
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path ${data_root}/laion_webdataset --save-path ${data_root}/laion_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$dataset" == "datacomp-dfn" ]; then
      mkdir ${data_root}/datacomp-medium/dfn_single_pkl_pil
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path ${data_root}/datacomp_medium_dfn_webdataset --save-path ${data_root}/datacomp_dfn_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    else
      mkdir ${data_root}/datacomp-medium/hq_single_pkl_pil
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path ${data_root}/datacomp_medium_mlm_filter_su_85_union_dfn_webdataset/ --save-path ${data_root}/datacomp_hq_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    fi
  } &
  i=$(($i + 1))
done

wait
end=`date +"%s"`
echo "time: " `expr $end - $start`
