# we do have test in ops_cpu.py: everything that will involve Conv2D layer
from tinygrad.tensor import Tensor
from tinygrad.ops.ops_cpu import Conv2D
from datasets import fetch_mnist
X_train, _, _, _ = fetch_mnist()


class Conv2D_mod(Conv2D):
  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    tx, tw, x_shape = ctx.saved_tensors
    _,rcout,cin,H,W = tw.shape
    ys,xs = ctx.stride
    OY,OX = x_shape[2:4]

    ggg = grad_output.reshape(bs,ctx.groups,rcout,oy,ox)

    gdw = np.zeros((ctx.groups,rcout,cin,H,W), dtype=tx.dtype)
    for g in range(ctx.groups):
      #'ikYX,ijYXyx -> kjyx'
      gdw[g] += np.tensordot(ggg[:,g], tx[:,g], ((0,2,3),(0,2,3)))

    # needs to be optimized
    gdx = np.zeros((bs,ctx.groups,cin,OY,OX), dtype=tx.dtype)
    # add path via np.einsum_path
    # path[0] is the algorithm to execute,
    # path[1] is the information of the calculated steps and time comsumption
    # path = ['einsum_path', (0, 1)] # via grid search, precached here for speed
    for k in range(oy*ox):
      Y, X = k//ox, k%ox
      iY,iX = Y*ys, X*xs
      # add path via einsum
      # gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw, optimize=path)

      # it's right but very slow: 37 sec
      # gdx[:,:,: , iY:iY+H, iX:iX+W] += np.tensordot(ggg[:,:,:,Y,X], tw, axes=2).sum(axis=0)

      for g in range(ctx.groups):
        tg = np.dot(ggg[:,g,:,Y,X].reshape(bs, -1), tw[g].reshape(rcout, -1))
        gdx[:, g, :, iY:iY+H, iX:iX+W] += tg.reshape((bs, cin, H, W))

    return gdx.reshape((bs, ctx.groups*cin, OY, OX)), gdw.reshape((ctx.groups*rcout, cin, H, W))

import unittest
class Test(unittest.TestCase):
  def test_conv(self):
    w = Tensor(3,3)
    # requires register ?
    # or, we can just modified the source code
    mod = Conv2D_mod()
    mod.forward(

