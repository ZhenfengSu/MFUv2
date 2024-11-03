
import torch
A = torch.randn(2,3)
B = torch.randn(3,5)

# Quantize the tensors to int8
quantized_A = torch.quantize_per_tensor(A, scale=1.0, zero_point=0, dtype=torch.qint8)
quantized_B = torch.quantize_per_tensor(B, scale=1.0, zero_point=0, dtype=torch.qint8)

# perform matrix multiplication
# quantized_A = quantized_A.to("cuda")
# quantized_B = quantized_B.to("cuda")
quantized_A = quantized_A.cuda()
quantized_B = quantized_B.cuda()
# result = torch.matmul(quantized_A, quantized_B)
result = torch._int_mm(quantized_A, quantized_B)
# result = quantized_A @ quantized_B

# Dequantize the result
dequantized_result = result.dequantize()
print(dequantized_result)
