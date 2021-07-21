#from mmocr.apis.inference import model_inference
from PIL import Image
from numpy import asarray
import torch
import tensorflow as tf
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmocr.models import build_detector

model_path=r'/home/ubuntu/Desktop/itr_forms_mmocr/mmocr/work_dirs/itr_output_mod_data/epoch_50.pth'
image_path=r'/home/ubuntu/Desktop/itr_forms_mmocr/mmocr/data/test_images/347334777_OB13961709_IncomeProof_28072020_32123_0.jpg'
cfg = Config.fromfile("configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt_old.py")
img=Image.open(image_path)
numpy_data=asarray(img)
data_tensor=tf.convert_to_tensor(numpy_data)
model = build_detector(cfg.model)
load_checkpoint(model, args.checkpoint, map_location='cpu')
model = MMDataParallel(model, device_ids=[0])
with torch.no_grad():
    result = model(return_loss=False, rescale=True, **data_tensor)
print("Results**********\n",result)
#print(type(img))
#dic=model_inference(model_path,numpy_data)
#print(dic)
