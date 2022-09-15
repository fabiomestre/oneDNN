
#***************************************************************************
#*  Copyright 2020 Codeplay Software Limited
#*  Licensed under the Apache License, Version 2.0 (the "License");
#*  you may not use this file except in compliance with the License.
#*  You may obtain a copy of the License at
#*
#*      http://www.apache.org/licenses/LICENSE-2.0
#*
#*  For your convenience, a copy of the License has been included in this
#*  repository.
#*
#*  Unless required by applicable law or agreed to in writing, software
#*  distributed under the License is distributed on an "AS IS" BASIS,
#*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#*  See the License for the specific language governing permissions and
#*  limitations under the License.
#*
#**************************************************************************

#! /usr/bin/env python3

import os
import re
import h5py as h5
import numpy as np
import urllib.request
from sys import argv

name_tensor_pairs = []

def parse_vgg16_tf(name):
    """
    Parses the key such that the result is of the form operator_index for the
    VGG16 model. This assumes TF-style tensors (that is, inputs are in NHWC
    format and weights in HWIO format).

    Keys are initially of the form:

      block{BL_IDX}_op({OP_IDX}) | fc{IDX} | flatten | predictions | input_1

    where op can be either 'conv' or 'pool'. The latter has no associated index.
    """
    # Ignore parameter-free layers.
    if 'input' in name or 'flatten' in name or 'pool' in name:
        return name

    # Remove redundant token at beginning.
    tokens = name.split('/')[1:]
    name = '_'.join(tokens)
    cleaned_key = re.sub('predictions', 'fc3', name)

    # Modify to allow easier matching with mapping dict above.
    cleaned_key = re.sub('W_1:0$', 'weights', cleaned_key)
    cleaned_key = re.sub('b_1:0$', 'biases', cleaned_key)

    # Swap the op and block tokens.
    if 'block' in cleaned_key:
        tokens = cleaned_key.split('_')
        block_token = tokens[0]
        op_token = tokens[1]
        operator_code = re.sub('[0-9]', '', tokens[1])
        operator_index = re.sub(operator_code, '', tokens[1])
        rearranged_tokens = [operator_code] + [operator_index] + [block_token[-1]] + tokens[2:]
        rearranged_key = '_'.join(rearranged_tokens)
        return rearranged_key
    else:
        return cleaned_key

def char_to_index(match):
    import string
    token = match.group(0)
    digit = re.search('[a-z]', token).group(0)
    digit_index = string.ascii_lowercase.index(digit)
    token = re.sub('[a-z]', '_' + str(digit_index+1), token)
    return token

def parse_resnet50_tf(name):
    """
    Parses the key such that the result is of the form operator_index for the
    VGG16 model. This assumes TF-style tensors (that is, inputs are in NHWC
    format and weights in HWIO format).

    Keys are initially of the form:

      block{BL_IDX}_op({OP_IDX}) | fc{IDX} | flatten | predictions | input_1

    where op can be either 'conv' or 'pool'. The latter has no associated index.
    """
    # Ignore parameter-free layers.
    if 'input' in name or 'flatten' in name or 'pool' in name:
        return name

    # Remove redundant token at beginning.
    tokens = name.split('/')[1:]
    name = '_'.join(tokens)
    cleaned_key = re.sub('predictions', 'fc3', name)

    # Modify to allow easier matching with mapping dict above.
    cleaned_key = re.sub('W:0$', 'weights', cleaned_key)
    cleaned_key = re.sub('b:0$', 'biases', cleaned_key)
    cleaned_key = re.sub('gamma:0$', 'scale', cleaned_key)
    cleaned_key = re.sub('beta:0$', 'shift', cleaned_key)
    cleaned_key = re.sub('running_mean:0$', 'mean', cleaned_key)
    cleaned_key = re.sub('running_std:0$', 'var', cleaned_key)

    # Swap the op and block tokens.
    if 'block' in cleaned_key:
        tokens = cleaned_key.split('_')
        block_token = tokens[0]
        op_token = tokens[1]
        operator_code = re.sub('[0-9]', '', tokens[1])
        operator_index = re.sub(operator_code, '', tokens[1])
        rearranged_tokens = [operator_code] + [operator_index] + [block_token[-1]] + tokens[2:]
        rearranged_key = '_'.join(rearranged_tokens)
        return rearranged_key
    if 'branch' in cleaned_key:
        return re.sub('(\d[a-z])', char_to_index, cleaned_key)
    else:
        return cleaned_key

def transform_hwio(name, tensor):
    if 'weights' in name and ('conv' in name or 'res' in name):
        ordering = (3,0,1,2)  # WHIO to OIWH, change to 3, 2, 1, 0 for WHIO -> OIHW
        tensor = np.transpose(tensor, ordering)
    return name, tensor


def func_with_objs(parser, tensor_transform):
    def f(name, obj):
        if ':0' in name:
            tensor = np.ndarray(obj.shape, dtype=np.float32)
            obj.read_direct(tensor)
            parsed_name = parser(name)
            parsed_name, tensor = tensor_transform(parsed_name, tensor)
            name_tensor_pairs.append((parsed_name, tensor))
    return f

def dump_data(name, parser, transform):
    print("Dumping and converting weights for " + name)
    f = h5.File(name, 'r')
    model_name = name.split('_')[0]
    need_transpose = 'tf' in name
    transpose_token = '_transposed' if need_transpose else ''

    f.visititems(func_with_objs(parser, transform))

    for (name, tensor) in name_tensor_pairs:
        name = name + '.bin'
        full_path = model_name + transpose_token + '_param_files/' + name
        if not os.path.isdir(model_name + transpose_token + '_param_files/'):
            os.mkdir(model_name + transpose_token + '_param_files/')
        print('filename: ' + full_path)
        with open(full_path, 'wb') as f_param:
            tensor.tofile(f_param)

def download_model(url,fname):
    if os.path.isfile(fname):
            print("Model file already found, skipping download step")
    else:
        print("Downloading model from " + url)
        urllib.request.urlretrieve(url, fname)

def download_convert_vgg16():
    url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    fname = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    download_model(url,fname)
    dump_data(fname, parser=parse_vgg16_tf, transform=transform_hwio)
    print("Removing model file " + fname)
    os.remove(fname)

def download_convert_resnet50():
    url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
    fname = "resnet50_weights_tf_dim_ordering_tf_kernels.h5"
    download_model(url,fname)
    dump_data(fname, parser=parse_resnet50_tf, transform=transform_hwio)
    print("Removing model file " + fname)
    os.remove(fname)

if __name__ == '__main__':
    for name in argv[1:]:
        if 'vgg16' in name:
            download_convert_vgg16()
        elif 'resnet50' in name:
            download_convert_resnet50()
        elif 'all' in name:
            download_convert_vgg16()
            download_convert_resnet50()
        else:
            raise ValueError('Unrecognised network')
        
        
        
