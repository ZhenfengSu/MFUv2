import torch

A = torch.randint(-128, 127, size=(2048, 1184),
                    device=0).to(torch.int8)
B = torch.randint(-128, 127, size=(1184, 512),
                    device=0).to(torch.int8)
res = torch._int_mm(A, B)
print(res)