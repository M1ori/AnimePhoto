import paddle
from ppgan.models.generators import AnimeGenerator

weight_path = 'output_dir/animeganv2-2025-06-07-23-22/epoch_30_weight.pdparams'

net = AnimeGenerator()
checkpoint = paddle.load(weight_path)
net.set_state_dict(checkpoint['netG'])
net.eval()

x_spec = paddle.static.InputSpec(shape = [1, 3, 256, 256],
                                 dtype = 'float32',
                                 name = 'x')

net = paddle.jit.to_static(net, input_spec = [x_spec])
paddle.jit.save(net, 'output_dir/inference/animegan')