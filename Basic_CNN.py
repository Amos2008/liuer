import torch
in_channels,out_channels = 5,10
width,height = 100,100
kernel_size =3
batch_size =5
input = torch.randn(batch_size,in_channels,width,height)

conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
# conv_layer.weight.data=kernel.data
output = conv_layer(input)
# print(input)
# print(input.shape)
# print(type(input),":",input.type())
#
# print(output.shape)
# print(type(output),":",output.type())
# print(conv_layer.weight.shape)


class Student:
    def __call__(self,param):
        print('被重写，相当于构造函数 ')
        print("传入参数".format(type(param),param))

        res=self.forward(param)
        return  res
    def forward(self,input_):
        print('forward被调用了')
        return input_

a = Student()
input_param=a('data')
print(input_param)
# a('data')=a.forward(data) 等价于神经网络的forward


