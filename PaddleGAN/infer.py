import cv2
import numpy as np
from paddle.inference import create_predictor
from paddle.inference import Config as PredictConfig

model_path = "./output_dir/inference/animegan.pdmodel"
params_path = "./output_dir/inference/animegan.pdiparams"
pred_cfg = PredictConfig(model_path, params_path)
pred_cfg.enable_memory_optim()
pred_cfg.switch_ir_optim(True)
pred_cfg.enable_use_gpu(500, 0)
predictor = create_predictor(pred_cfg)

input_names = predictor.get_input_names()
input_handle = {}
for i in range(len(input_names)):
    input_handle[input_names[i]] = predictor.get_input_handle(input_names[i])
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])

img = cv2.imread("a.png", flags = cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
img = img.astype(np.float32)

img[:, :, 0] -= -4.4346957
img[:, :, 1] -= -8.665916
img[:, :, 2] -= 13.100612

img = img / 127.5 - 1
img = np.transpose(img[np.newaxis, :, :, :], [0, 3, 1, 2])

input_handle["x"].copy_from_cpu(img)
predictor.run()
results = output_handle.copy_to_cpu()

results = results.squeeze()
cartoon = np.transpose(results, (1, 2, 0))
cartoon = (cartoon + 1) * 127.5
cartoon = cartoon.astype("uint8")
cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
cv2.imwrite("./output_dir/result3.jpg", cartoon)