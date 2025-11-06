import sys, os, glob, nibabel as nib, numpy as np

def dice(a,b):
    a=a>0; b=b>0
    return 2*np.sum(a & b)/(np.sum(a)+np.sum(b)+1e-8)

root = sys.argv[1]   # e.g. data/my-ct-dataset/val
scores=[]
for case in glob.glob(os.path.join(root,'*')):
    pred=os.path.join(case,'mask_predicted.nii')
    gt  =os.path.join(case,'mask.nii')
    if os.path.exists(pred) and os.path.exists(gt):
        d = dice(nib.load(pred).get_fdata(),
                 nib.load(gt).get_fdata())
        scores.append(d)
        print(os.path.basename(case), f"Dice={d:.3f}")
print("Mean Dice:", np.mean(scores))