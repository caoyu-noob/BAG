# BAG
Implementation for NAACL-2019 paper 

**BAG: Bi-directional Attention Entity Graph Convolutional Network forMulti-hop Reasoning Question Answering**

![BAG Framework](https://github.com/caoyu1991/BAG/blob/master/BAG.png)

## Requirement
1. Python 3.6
2. TensorFlow >= 1.11.0
3. Pytorch >= 0.4.1
4. SpaCy >= 2.0.12 (You need to install "en" module via )
5. allennlp >= 0.7.1
6. nltk >= 3.3

And some other packages

I run it using two NVIDIA GTX1080Ti GPUs each one has 11GB memory. To run it with 
default batch size 32, at least 16GB memory is needed.

## How to run
- Before run

You need to download pretrained 840B 300d [GLoVe embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip), 
and pretrained original size ELMo embedding [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
 and [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json) and put them under directory _/data_. 
 
- Preprocessing dataset

You need to download [QAngaroo WIKIHOP dataset](https://drive.google.com/file/d/1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA/view)
, unzip it and put the json files under the root directory. Then run prerpocessing script 

`python prepro.py {json_file_name}`

It will generate four preprocessed pickle file in the root directory which will be used
in the training or prediction.

- Train the model

Train the model using following command which will follow the configure
in original paper

`python BAG.py {train_json_file_name} {dev_json_file_name} --use_multi_gpu=true`

Please make sure you have run preprocessing for both train file and dev
file before training. And please make sure you have CUDA0 and CUDA1 available.
If you have single GPU with more than 16GB memory, you can remove parameter 
_--use_multi_gpu_.

- Predict 

After training it will put trained model onto directory _/models_.
You can predict the answer of a json file using following command 

`python BAG.py {predict_json_file_name} {predict_json_file_name} --use_multi_gpu=true --evaluation_mode=true`