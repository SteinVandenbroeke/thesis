# Editable
dataset=cub_val
result_files=result_cub/BESTIE_CUB.json,result_cub/CIM-CUB.json
save_dir=./vis_val_$(date +%Y%m%d_%H%M%S)_${dataset}

##############
# Not editable
# train CIM
source .run_venv/bin/activate\

python ./visualize/WSIS_metric_analyse.py --dataset ${dataset} \
--result_files ${result_files} \
--save_dir ${save_dir}

############