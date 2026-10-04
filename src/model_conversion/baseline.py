import time
import os, sys
import torch
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights
ROOT_DIR=Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


from model_conversion.config.config import ARTIFACT_PATH as PATH_DIR
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

weights=ResNet18_Weights.DEFAULT
model=resnet18(weights=weights)
model.eval()
model.to(device)
x=torch.randn(
    1,3,224,224,device=device
)


torch.save(model.state_dict(),PATH_DIR/"resnet18.pth")
print("Model artifact size:",os.path.getsize(PATH_DIR/"resnet18.pth")/(1024**2),"MB")

start=time.perf_counter()
loaded_model=resnet18(weights=None)
state_dict=torch.load(PATH_DIR/"resnet18.pth",map_location=device)

loaded_model.load_state_dict(state_dict)

loaded_model.eval()
loaded_model.to(device)
if device.type=="cuda":
    torch.cuda.synchronize()

load_time=time.perf_counter()-start
print("Load time:",load_time,"seconds")

with torch.inference_mode():
    for _ in range(20):
        output=loaded_model(x)
        
if device.type=="cuda":
    torch.cuda.synchronize()
    
iterations=100
with torch.inference_mode():
    for _ in range(iterations):
        output=loaded_model(x)
        
if device.type=="cuda":
    torch.cuda.synchronize()
    
elapsed=time.perf_counter()-start
latency=elapsed/iterations
throughput=iterations/elapsed

print("\n ------ BASELINE ------")
print("Latency:", latency*1000,"ms")
print("Throughput:",throughput,"samples/sec")

print("\nOutput Shape:",output.shape)
print(output[0,:5])
