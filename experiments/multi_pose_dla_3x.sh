cd src
# train
python main.py multi_pose --exp_id my_dla_3x --dataset coco_hp --batch_size 4 \
  --lr 1e-4 --gpus 0 --num_workers 8 --num_epochs 320 --lr_step 270,300  \
  --load_model ../models/multi_pose_dla_3x.pth

# or use the following command if your have dla_1x trained
# python main.py multi_pose --exp_id dla_3x --dataset coco_hp --batch_size 128 --master_batch 9 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --load_model ../exp/multi_pose/dla_1x/model_90.pth --resume
# test
python test.py multi_pose --exp_id dla_3x --dataset coco_hp --keep_res --resume
# flip test
python test.py multi_pose --exp_id dla_3x --dataset coco_hp --keep_res --resume --flip_test
cd ..
