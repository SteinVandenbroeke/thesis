python vis_example.py \
  -o "visualizations/BAS_basic_splitting_example.png" \
  -r "result_coco/CIM-COCO-val.json" \
  -v "000000478420:0SMG 000000478420:0SMG 000000478420:0CMG \n 000000476514:0SMG 000000476514:0SMG 000000476514:0CMG \n 000000297353:0SMG 000000297353:0SMG 000000297353:0CMG" \
  -t "Cross-Dataset Model Instance Segmentation Comparison" -c "person"


python vis_example.py \
  -o "visualizations/COCO_examples.png" \
  -r "result_coco/CIM-COCO-val.json" \
  -v "U:0SMG U:0SMG :0SMG U:0SMG \n U:0SMG U:0SMG U:0SMG U:0SMG \n U:0SMG U:0SMG U:0SMG U:0SMG\n " \
  -t "COCO GT dataset segmentation masks"

python vis_example.py \
  -o "visualizations/CUB_examples.png" \
  -r "result_cub/CIM-CUB.json" \
  -v "U:0SMG U:0SMG :0SMG U:0SMG \n U:0SMG U:0SMG U:0SMG U:0SMG \n U:0SMG U:0SMG U:0SMG U:0SMG\n " \
  -t "CUB GT dataset segmentation masks"

python vis_example.py \
  -o "visualizations/VOC_examples.png" \
  -r "result_voc/CIM-VOC-val.json" \
  -v "U:0SMG U:0SMG :0SMG U:0SMG \n U:0SMG U:0SMG U:0SMG U:0SMG \n U:0SMG U:0SMG U:0SMG U:0SMG\n " \
  -t "VOC GT dataset segmentation masks"

python vis_example.py \
  -o "visualizations/COCO_bbox_pix_boundry_examples.png" \
  -r "result_voc/BAS_COCO.json" \
  -v "33707:0SMG" \
  -t "VOC GT dataset segmentation masks"

  python vis_example.py \
  -o "visualizations/BAS_result_examples.png" \
  -r "result_voc/BAS.json" \
  -v "2010_004348:0SMG 2010_004348:0SMP" \
  -t "VOC GT dataset segmentation masks"

  python regenerate_plots.py --csvs vis_val_20260816_025006_voc_val/combined_occlusion_impact_bars_Any_Cla
ss_plot.csv vis_val_20260816_025006_voc_val/combined_occlusion_impact_bars_Same_Class_plot.csv --output visualizations/combined_occlusion_impact_bars_Side_By_Side_BAS_WSSS.png


  python vis_example.py \
  -o "visualizations/WSSS_WSIS_difference_example_BAS_VOC.png" \
  -r "result_voc/BAS.json,result_voc/BAS_WSSS.json" \
  -v "2007_000925:0SMG 2007_000925:0SMP 2007_000925:1SMP" \
  -t "Comparison between WSSS and WSIS mask prediction"

  python vis_example.py \
  -o "visualizations/BAS_VOC_Combined_by_occlusion.png" \
  -r "result_voc/BAS.json" \
  -v "2010_003293:0SMG 2010_003293:0SMP" \
  -t "BAS VOC Combined by occlusion score"



  python vis_example.py \
  -o "visualizations/BAS_image_examples_VOC.png" \
  -r "result_voc/BAS.json" \
  -v "2007_005469:0SMG 2007_005469:0SMP  2009_004942:0SMG 2009_004942:0SMP  2007_005304:0SMG 2007_005304:0SMP
  \n  2007_009687:0SMG 2007_009687:0SMP  2007_003711:0SMG 2007_003711:0SMP  2008_000673:0SMG 2008_000673:0SMP
  \n  2010_000159:0SMG 2010_000159:0SMP  2010_003381:0SMG 2010_003381:0SMP  2007_009323:0SMG 2007_009323:0SMP
  \n  2007_000636:0SMG 2007_000636:0SMP  2007_000830:0SMG 2007_000830:0SMP  2007_009897:0SMG 2007_009897:0SMP" \
  -t "BAS image examples VOC"

python vis_example.py \
  -o "visualizations/BAS_image_examples_COCO.png" \
  -r "result_coco/BAS_WSIS.json" \
  -v "000000000285:0SMG 000000000285:0SMP  000000005001:0SMG 000000005001:0SMP  000000006040:0SMG 000000006040:0SMP
  \n  000000006954:0SMG 000000006954:0SMP  000000010363:0SMG 000000010363:0SMP  000000012667:0SMG 000000012667:0SMP
  \n  000000433243:0SMG 000000433243:0SMP  000000020247:0SMG 000000020247:0SMP  000000452122:0SMG 000000452122:0SMP
  \n  000000507042:0SMG 000000507042:0SMP  000000540414:0SMG 000000540414:0SMP  000000572678:0SMG 000000572678:0SMP" \
  -t "BAS image examples COCO"


  python vis_example.py \
  -o "visualizations/BESTIE_image_examples_VOC.png" \
  -r "result_voc/BESTIE_VOC.json" \
  -v "2007_005469:0SMG 2007_005469:0SMP  2009_004942:0SMG 2009_004942:0SMP  2007_005304:0SMG 2007_005304:0SMP
  \n  2007_009687:0SMG 2007_009687:0SMP  2007_003711:0SMG 2007_003711:0SMP  2008_000673:0SMG 2008_000673:0SMP
  \n  2010_000159:0SMG 2010_000159:0SMP  2010_003381:0SMG 2010_003381:0SMP  2007_009323:0SMG 2007_009323:0SMP
  \n  2007_000636:0SMG 2007_000636:0SMP  2007_000830:0SMG 2007_000830:0SMP  2007_009897:0SMG 2007_009897:0SMP" \
  -t "BESTIE image examples VOC"


python vis_example.py \
  -o "visualizations/BESTIE_SUCCES_result_examples.png" \
  -r "result_voc/BESTIE_VOC.json" \
  -v "2010_001646:0SMG 2010_001646:0SMP" \
  -t "VOC Success full instance split"

 python instance_scatter_plot_gen.py --dataset voc_val \
 --result_files result_voc/BESTIE_VOC.json \
--save_dir visualizations


  python vis_example.py \
  -o "visualizations/CIM_image_examples_VOC.png" \
  -r "result_voc/CIM-VOC-val.json" \
  -v "2007_005469:0SMG 2007_005469:0SMP  2009_004942:0SMG 2009_004942:0SMP  2007_005304:0SMG 2007_005304:0SMP
  \n  2007_009687:0SMG 2007_009687:0SMP  2007_003711:0SMG 2007_003711:0SMP  2008_000673:0SMG 2008_000673:0SMP
  \n  2010_000159:0SMG 2010_000159:0SMP  2010_003381:0SMG 2010_003381:0SMP  2007_009323:0SMG 2007_009323:0SMP
  \n  2007_000636:0SMG 2007_000636:0SMP  2007_000830:0SMG 2007_000830:0SMP  2007_009897:0SMG 2007_009897:0SMP" \
  -t "CIM image examples VOC"

python vis_example.py \
  -o "visualizations/CIM_image_examples_COCO.png" \
  -r "result_coco/CIM-COCO-val.json" \
  -v "000000000285:0SMG 000000000285:0SMP  000000005001:0SMG 000000005001:0SMP  000000006040:0SMG 000000006040:0SMP
  \n  000000006954:0SMG 000000006954:0SMP  000000010363:0SMG 000000010363:0SMP  000000012667:0SMG 000000012667:0SMP
  \n  000000433243:0SMG 000000433243:0SMP  000000020247:0SMG 000000020247:0SMP  000000452122:0SMG 000000452122:0SMP
  \n  000000507042:0SMG 000000507042:0SMP  000000540414:0SMG 000000540414:0SMP  000000572678:0SMG 000000572678:0SMP" \
  -t "CIM image examples COCO"


    python vis_example.py \
  -o "visualizations/CIM_VOC_Failed_Instance_Split.png" \
  -r "result_voc/CIM-VOC-val.json" \
  -v "2010_002868:0SMG 2010_002868:0SMP" \
  -t "CIM VOC Failed Instance Split"

   python instance_scatter_plot_gen.py --dataset voc_val \
 --result_files result_voc/BESTIE_VOC.json \
--save_dir visualizations

   python instance_scatter_plot_gen.py --dataset cub_val \
 --result_files result_cub/BESTIE_CUB.json \
--save_dir visualizations