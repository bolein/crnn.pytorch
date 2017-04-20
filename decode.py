from __future__ import print_function
import models.crnn as crnn
import torch
from torch.autograd import Variable
from torch import optim
import dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256, ngpu=1)
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))
model.eval()

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')

# create white noize image with same dimensions as input
x = dataset.get_white_noise_image(image.size).convert('L')

# pre process input
image = transformer(image)
x     = transformer(x)
image = image.view(1, *image.size())
x     = x.view(1, *x.size())

# wrap images into PyTorch variables
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
image = Variable(image)
x     = Variable(x, requires_grad=True)


# plot graphs
plt.ion()
plt.figure().suptitle('Loss')
ax = plt.gca()
line, = plt.plot([], [])
plt.figure().suptitle('Target image')
im_plot = plt.imshow(np.asarray(dataset.image_from_tensor(x.data[0])), cmap='gray')


def update_plot(epoch, loss, img):
    # update line
    line.set_xdata(np.append(line.get_xdata(), epoch))
    line.set_ydata(np.append(line.get_ydata(), loss))
    ax.relim()
    ax.autoscale_view()

    # update img
    im_plot.set_data(np.asarray(img))

    plt.draw()
    plt.pause(0.1)


# Do the network decoding
learning_rate = 0.01
loss_function = torch.nn.MSELoss()
optimizer = optim.Adam([x], lr=1e-2)

for epoch in range(500):
    # calculate loss of the output activations
    # we detach the target from the computational graph because we don't need
    # to compute the gradient for it
    loss = loss_function(model.forward_depth(x, depth=6), model.forward_depth(image, depth=6).detach())

    # Manually zero the gradients before running the backward pass
    if x.grad is not None:
        x.grad.data.zero_()

    # Use autograd to compute the backward pass.
    loss.backward()

    # update parameters using gradient descent
    # x.sub_(learning_rate * x.grad)
    optimizer.step()

    print('epoch {} loss: {}'.format(epoch, loss.data[0]))
    update_plot(epoch, loss.data[0], dataset.image_from_tensor(x.data[0]))


# original = dataset.image_from_tensor(image.data[0])
# original.show()
result = dataset.image_from_tensor(x.data[0])
result.show()

# converter = utils.strLabelConverter(alphabet)
#
# preds = model.forward(x)
#
# _, preds = preds.max(2)
# preds = preds.squeeze(2)
# preds = preds.transpose(1, 0).contiguous().view(-1)
#
# preds_size = Variable(torch.IntTensor([preds.size(0)]))
# raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
# sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
# print('%-20s => %-20s' % (raw_pred, sim_pred))
