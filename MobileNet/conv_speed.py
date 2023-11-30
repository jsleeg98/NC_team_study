import torch
import time

batch = 128
inps = [16, 32, 64, 128]
oups = [16, 32, 64, 128]

for inp in inps:
    for oup in oups:
        inp = inp
        oup = oup
        device = 'cuda'
        print('-' * 30)
        print(f'batch : {batch}')
        print(f'inp : {inp}')
        print(f'oup : {oup}')


        standard_conv = torch.nn.Conv2d(inp, oup, groups=1, kernel_size=(3, 3), padding=1, stride=1, bias=False)

        depthwise_conv = torch.nn.Conv2d(inp, inp, groups=inp, kernel_size=(3, 3), padding=1, stride=1, bias=False)
        pointwise_conv = torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False)


        input = torch.rand(batch, inp, 224, 224)
        input = input.to(device)
        standard_conv.to(device)
        depthwise_conv.to(device)
        pointwise_conv.to(device)

        start = time.time()
        for i in range(1000):
            standard_conv(input)
        print(f'standard_conv (1000 iter) : {time.time() - start}')

        start = time.time()
        for i in range(1000):
            tmp = depthwise_conv(input)
            pointwise_conv(tmp)
        print(f'depthwise_separable_conv (1000 iter) : {time.time() - start}')
        print('-' * 30)