import os, argparse
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
from utils.utils import select_device, model_fuse

def export_torchscript(opt, model, img, prefix='TorchScript'):
    print('Starting TorchScript export with pytorch %s...' % torch.__version__)
    f = os.path.join(opt.save_path, 'best.ts')
    ts = torch.jit.trace(model, img, strict=False)
    ts.save(f)
    print(f'Export TorchScript Model Successfully.\nSave sa {f}')

def export_onnx(opt, model, img, prefix='ONNX'):
    import onnx
    f = os.path.join(opt.save_path, 'best.onnx')
    print('Starting ONNX export with onnx %s...' % onnx.__version__)
    if opt.dynamic:
        dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'}, 'output':{0: 'batch'}}
    else:
        dynamic_axes = None
    
    torch.onnx.export(
        (model.to('cpu') if opt.dynamic else model), 
        (img.to('cpu') if opt.dynamic else img),
        f, verbose=False, opset_version=13, input_names=['images'], output_names=['output'], dynamic_axes=dynamic_axes)
    
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    if opt.simplify:
        try:
            import onnxsim
            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
        onnx.save(onnx_model, f)
    
    print(f'Export Onnx Model Successfully.\nSave sa {f}')

def export_engine(opt, model, img, workspace=4, prefix='TensorRT'):
    export_onnx(opt, model, img)
    onnx_file = os.path.join(opt.save_path, 'best.onnx')
    assert img.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    import tensorrt as trt
    print('Starting TensorRT export with TensorRT %s...' % trt.__version__)
    f = os.path.join(opt.save_path, 'best.engine')

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if opt.verbose else trt.Logger()
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(str(onnx_file)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_file}')
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input {inp.name} with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        print(f'output {out.name} with shape {out.shape} and dtype {out.dtype}')
    
    if opt.dynamic:
        if img.shape[0] <= 1:
            print(f"{prefix} WARNING: --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *img.shape[1:]), (max(1, img.shape[0] // 2), *img.shape[1:]), img.shape)
        config.add_optimization_profile(profile)

    print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and opt.half else 32} engine in {f}')
    if builder.platform_has_fast_fp16 and opt.half:
            config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    print(f'Export TensorRT Model Successfully.\nSave sa {f}')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--image_channel', type=int, default=3, help='image channel')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX batchsize')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--half', action="store_true", help='FP32 to FP16')
    parser.add_argument('--verbose', action="store_true", help='TensorRT:verbose export log')
    parser.add_argument('--export', default='torchscript', type=str, choices=['onnx', 'torchscript', 'tensorrt'], help='export type')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_known_args()[0]
    if not os.path.exists(os.path.join(opt.save_path, 'best.pt')):
        raise Exception('best.pt not found. please check your --save_path folder')
    DEVICE = select_device(opt.device)
    if opt.half:
        assert DEVICE.type != 'cpu', '--half only supported with GPU export'
        assert not opt.dynamic, '--half not compatible with --dynamic'
    ckpt = torch.load(os.path.join(opt.save_path, 'best.pt'))
    model = ckpt['model'].float().to(DEVICE)
    model_fuse(model)
    img = torch.rand((opt.batch_size, opt.image_channel, opt.image_size, opt.image_size)).to(DEVICE)

    return opt, (model.half() if opt.half else model), (img.half() if opt.half else img), DEVICE

if __name__ == '__main__':
    opt, model, img, DEVICE = parse_opt()

    if opt.export == 'onnx':
        export_onnx(opt, model, img)
    elif opt.export == 'torchscript':
        export_torchscript(opt, model, img)
    elif opt.export == 'tensorrt':
        export_engine(opt, model, img)