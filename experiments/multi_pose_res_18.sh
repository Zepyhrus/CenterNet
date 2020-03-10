cd src

python main.py multi_pose --exp_id my_res_18 --arch res_18 --dataset coco_hp --batch_size 16 \
  --lr 1e-3 --gpus 0 --num_workers 8 --num_epochs 320 --lr_step 240,300


# test
python test.py multi_pose --exp_id my_res_18  --arch res_18 --load_model ../exp/multi_pose/my_res_18/model_last.pth

# demo
python demo.py multi_pose --arch res_18 --demo ../person/png_img --load_model ../exp/multi_pose/my_res_18/model_last.pth


