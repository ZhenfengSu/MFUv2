import torch
from torch.utils import benchmark

def test_1():
    typ = torch.float16  #数据精度
    bs = 16384
    n = 4096
    a = torch.randn(bs, n).type(typ).cuda()
    b = torch.randn(n, n).type(typ).cuda()
    t = benchmark.Timer(
        stmt='a @ b',
        globals={'a': a, 'b': b})

    x = t.timeit(50)
    print(2*bs*n**2 / x.median /1e12)

def test_2():
    typ = torch.float32
    n = 1024 * 16
    batch = 4
    s = 100 # tokens per sequence
    a = torch.randn(batch,s,n).type(typ).cuda()
    b = torch.randn(n, n).type(typ).cuda()
    
    t = benchmark.Timer(
        stmt='a @ b',
        globals={'a': a, 'b': b})
    num_runs = 50  
    times = t.repeat(repeat=num_runs, number=1) 
    # Discard the first result  
    times = times[1:] 
    # breakpoint()
    x = t.timeit(50)
    print(2*batch*s*n**2 / x.median /1e12)

if __name__ == '__main__':
    test_1()