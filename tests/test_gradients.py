import torch
from torch.autograd import Variable

x = Variable(torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[4, 6, 9], [5, 6, 7], [7, 8, 9]]]).view(1, 2, 3, 3),requires_grad=True)

diff_y = x[:, :, 1:] - x[:, :, :-1]             # vertical gradient
diff_x = x[:, :, :, 1:] - x[:, :, :, :-1]       # horizontal gradient

print(diff_x)
print(diff_y)

h_x = x.size()[2]
w_x = x.size()[3]
print(h_x, w_x)

h_tv = x[:,:,1:,:]-x[:,:,:h_x-1,:]      # vertical gradient
w_tv = x[:,:,:,1:]-x[:,:,:,:w_x-1]      # horizontal gradient

print(w_tv)
print(h_tv)

def tensor_size(t):
    print(t.size()[1], t.size()[2], t.size()[3])
    return t.size()[1] * t.size()[2] * t.size()[3]

count_h = tensor_size(x[:, :, 1:, :])   # number of elements in the vertical gradient
count_w = tensor_size(x[:, :, :, 1:])   # number of elements in the horizontal gradient

print(count_h)
print(count_h)