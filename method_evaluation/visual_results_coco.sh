# Editable
dataset=coco_val
result_files=result_coco/BAS_WSSS.json,result_coco/BAS_WSIS.json,result_coco/CIM-COCO-val.json #,result_coco/BAS_COCO.json
#result_files=result_coco/CIM-COCO-val.json #,result_coco/BAS_COCO.json
#result_files=result_coco/coco_annotations_gpu_0.json
#result_files=result_coco/BAS_WSIS.json,result_coco/CIM-COCO-val.json
#result_files=result_coco/BAS_WSIS.json,result_coco/BAS_WSSS.json
#result_files="result_coco/BESTIE_COCO(4).json"
save_dir=./vis_val_$(date +%Y%m%d_%H%M%S)_${dataset}

##############
# Not editable
# train CIM
source .run_venv/bin/activate \

python ./visualize/WSIS_metric_analyse.py --dataset ${dataset} \
--result_files ${result_files} \
--save_dir ${save_dir}

############