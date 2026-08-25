# Editable
dataset=voc_val
result_files=result_voc/CIM-VOC-val.json,result_voc/BESTIE_VOC.json,result_voc/BAS_WSSS.json,result_voc/BAS_WSIS.json
#result_files=result_voc/BAS_WSSS.json,result_voc/BAS.json
result_files=result_voc/CIM-VOC-val.json
save_dir=./vis_val_$(date +%Y%m%d_%H%M%S)_${dataset}

##############
# Not editable
# train CIM
source .run_venv/bin/activate\

python ./visualize/WSIS_metric_analyse.py --dataset ${dataset} \
--result_files ${result_files} \
--save_dir ${save_dir}

############