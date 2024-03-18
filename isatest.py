# export DNNL_VERBOSE=1
# export DNNL_MAX_CPU_ISA=AVX2_VNNI_2

import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
 
if __name__ == '__main__':
   x = nn.Sequential(nn.Conv2d(3, 64, (3, 3)), nn.MaxPool2d(1,1))
   inp = torch.randn(1,3,112,112)
   traced = torch.jit.trace(x, inp)
   opt = torch.jit.optimize_for_inference(traced)
   # Run twice
   out = opt(inp)
   out = opt(inp)
