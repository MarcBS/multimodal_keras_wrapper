# Tutorial

## Basic components

There are two basic components that have to be built in order to use the Multimodal Keras Wrapper,
which are a **[Dataset](https://github.com/MarcBS/multimodal_keras_wrapper/blob/6d0b11248fd353cc189f674dc30beaf9689da182/keras_wrapper/dataset.py#L331)** and a **[Model_Wrapper](https://github.com/MarcBS/multimodal_keras_wrapper/blob/6d0b11248fd353cc189f674dc30beaf9689da182/keras_wrapper/cnn_model.py#L154)**.

The class **Dataset** is in charge of:
- Storing, preprocessing and loading any kind of data for training a model (inputs).
- Storing, preprocessing and loading the ground truth associated to our data (outputs).
- Loading the data in batches for training or prediction.

The Datasets can manage different [types of input/output data](https://github.com/MarcBS/multimodal_keras_wrapper/blob/6d0b11248fd353cc189f674dc30beaf9689da182/keras_wrapper/dataset.py#L389-L390), which can be summarized as:
- input types: 'raw-image', 'video', 'image-features', 'video-features', 'text'
- output types: 'categorical', 'binary', 'real', 'text', '3DLabel'

Currently, the class Dataset can be used for multiple kinds of multimodal problems,
e.g. image/video classification, detection, multilabel prediction, regression, image/video captioning, 
visual question answering, multimodal translation, neural machine translation, etc.

The class **Model_Wrapper** is in charge of:
- Storing an instance of a Keras' model.
- Receiving the inputs/outputs of the class Dataset and using the model for training or prediction.
- Providing two different methods for prediction. Either [predictNet()](http://marcbs.github.io/multimodal_keras_wrapper/modules.html#keras_wrapper.cnn_model.Model_Wrapper.predictNet), which uses a conventional Keras model for prediction, or [predictBeamSearchNet()](http://marcbs.github.io/multimodal_keras_wrapper/modules.html#keras_wrapper.cnn_model.Model_Wrapper.predictBeamSearchNet), which applies a BeamSearch for sequence generative models and additionally allows to create separate models **model_init** and **model_next** for applying an optimized prediction (see [this](https://github.com/MarcBS/multimodal_keras_wrapper/blob/b348ce9d52404434b1e98316c7f09b5d5fd3df00/keras_wrapper/cnn_model.py#L1319-L1328) and [this](https://github.com/MarcBS/multimodal_keras_wrapper/blob/f269207a65bfc77d5c2c89ea708bad8bff7f72ab/keras_wrapper/cnn_model.py#L1057) for further information). 

In this tutorial we will learn how to create each of the two basic components and how use a
model for training and prediction.


## Creating a Dataset

First, let's create a simple Dataset object with some sample data. 
The data used for this example can be found in `/repository_root/data/sample_data`.


Dataset parameters definition.

```
from keras_wrapper.dataset import Dataset

dataset_name = 'test_dataset'
image_id = 'input_image'
label_id = 'output_label'
images_size = [256, 256, 3]
images_crop_size = [224, 224, 3]
train_mean = [103.939, 116.779, 123.68]
base_path = '</absolute/path/to/multimodal_keras_wrapper>/data/sample_data'
```

Empty dataset instance creation

```
ds = Dataset(dataset_name, base_path+'/images')
```


Insert dataset/model inputs

```
# train split
ds.setInput(base_path + '/train.txt', 'train',
            type='raw-image', id=image_id,
            img_size=images_size, img_size_crop=images_crop_size)
# val split
ds.setInput(base_path + '/val.txt', 'val',
            type='raw-image', id=image_id,
            img_size=images_size, img_size_crop=images_crop_size)
# test split
ds.setInput(base_path + '/test.txt', 'test',
            type='raw-image', id=image_id,
            img_size=images_size, img_size_crop=images_crop_size)
```

Insert pre-calculated images train mean

```
ds.setTrainMean(train_mean, image_id)
```

Insert dataset/model outputs

```
# train split 
ds.setOutput(base_path+'/train_labels.txt', 'train',
           type='categorical', id=label_id)
# val split
ds.setOutput(base_path+'/val_labels.txt', 'val',
           type='categorical', id=label_id)
# test split        
ds.setOutput(base_path+'/test_labels.txt', 'test',
           type='categorical', id=label_id)
```

## Saving or loading a Dataset

```
from keras_wrapper.dataset import saveDataset, loadDataset

save_path = '</absolute/path/to/multimodal_keras_wrapper>/Datasets'

# Save dataset
saveDataset(ds, save_path)

# Load dataset
ds = loadDataset(save_path+'/Dataset_'+dataset_name+'.pkl')
```

In addition, we can print some basic information of the data stored in the dataset:

```
print ds
```

## Creating a Model_Wrapper

Model_Wrapper parameters definition.

```
from keras_wrapper.cnn_model import Model_Wrapper

model_name = 'our_model'
type = 'VGG_19_ImageNet'
save_path = '</absolute/path/to/multimodal_keras_wrapper>/Models/'
```

Create a basic CNN model

```
net = Model_Wrapper(nOutput=2, type=type, model_name=model_name, input_shape=images_crop_size)
net.setOptimizer(lr=0.001, metrics=['accuracy']) # compile it
```

By default, the model type built is the one defined in [Model_Wrapper.basic_model()](https://github.com/MarcBS/multimodal_keras_wrapper/blob/6d0b11248fd353cc189f674dc30beaf9689da182/keras_wrapper/cnn_model.py#L2003).
Although, any kind of custom model can be defined just by:
- Defining a new method for the class Model_Wrapper which builds the model and stores it in self.model.
- Referencing it with type='method_name' when creating a new Model_Wrapper instance.


## Saving or loading a Model_Wrapper

```
from keras_wrapper.cnn_model import saveModel, loadModel

save_epoch = 0

# Save model
saveModel(net, save_epoch)

# Load model
net = loadModel(save_path+'/'+model_name, save_epoch)
```


## Connecting a Dataset to a Model_Wrapper

In order to provide a correct communication between the Dataset and the Model_Wrapper objects, we have to provide the links between the Dataset ids positions and their corresponding layer identifiers in the Keras' Model as a dictionary.

In this case we only have one input and one output, for this reason both ids are mapped to position 0 of our Dataset.

```
net.setInputsMapping({net.ids_inputs[0]: 0})
net.setOutputsMapping({net.ids_outputs[0]: 0})
```


## Training

We can specify several options for training our model, which are [summarized here](http://marcbs.github.io/multimodal_keras_wrapper/modules.html#keras_wrapper.cnn_model.Model_Wrapper.trainNet). If any of them is overriden then the [default values](https://github.com/MarcBS/multimodal_keras_wrapper/blob/011393580b2253a01c168d638b8c0bd06fe6d522/keras_wrapper/cnn_model.py#L454-L458) will be used.

```
train_overriden_parameters = {'n_epochs': 2, 'batch_size': 10}

net.trainNet(ds, train_overriden_parameters)
```

## Prediction

The same applies to the prediction method. We can find the [available parameters here](http://marcbs.github.io/multimodal_keras_wrapper/modules.html#keras_wrapper.cnn_model.Model_Wrapper.predictNet) and the [default values here](https://github.com/MarcBS/multimodal_keras_wrapper/blob/011393580b2253a01c168d638b8c0bd06fe6d522/keras_wrapper/cnn_model.py#L1468-L1470).

```
predict_overriden_parameters = {'batch_size': 10, 'predict_on_sets': ['test']}

net.predictNet(ds, predict_overriden_parameters)
```