# pituitary-gland

python crop_head_dataset_headanchored.py data\my-ct-dataset data\my-ct-dataset-cropped 144 96 288 bottom 8

python 1_create_dataset.py data\config\default.yaml data\my-ct-dataset-cropped data\my-ct-mv.h5

python 2_train_segmentation.py data\config\default.yaml data\my-ct-mv.h5 runs\

for /D %C in ("data\my-ct-dataset-cropped\val\*") do python 5_predict.py "data\config\default.yaml" "runs" "%~fC" "%~fC\mask_predicted.nii"

python eval_dice.py "data\my-ct-dataset-cropped\val"
