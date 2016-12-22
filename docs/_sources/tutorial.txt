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
base_path = '/absolute/path/to/multimodal_keras_wrapper/data/sample_data'
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

## Saving or Loading a Dataset

```
from keras_wrapper.dataset import saveDataset, loadDataset

save_path = '/absolute/path/to/multimodal_keras_wrapper/Datasets'

# Save dataset
saveDataset(ds, save_path)

# Load dataset
ds = loadDataset(save_path+'/Dataset_'+dataset_name+'.pkl')
```


## Creating a Model_Wrapper

Model_Wrapper parameters definition.

```
from keras_wrapper.cnn_model import Model_Wrapper

model_name = 'our_model'
type = 'basic_model'
save_path = '/absolute/path/to/multimodal_keras_wrapper/Models/'
```

Create a basic CNN model

```
net = Model_Wrapper(nOutput=2, type=type, model_name=model_name, input_shape=images_crop_size)
```

By default, the model type built is the one definied in [Model_Wrapper.basic_model()](https://github.com/MarcBS/multimodal_keras_wrapper/blob/6d0b11248fd353cc189f674dc30beaf9689da182/keras_wrapper/cnn_model.py#L2003).
Although, any kind of custom model can be defined just by:
- Defining a new method for the class Model_Wrapper which builds the model and stores it in self.model.
- Referencing it with type='method_name' when creating a new Model_Wrapper instance.


## Saving or Loading a Model_Wrapper

```
from keras_wrapper.cnn_model import saveModel, loadModel

save_epoch = 0

# Save model
saveModel(net, save_epoch)

# Load model
net = loadModel(save_path+'/'+model_name, save_epoch)
```


## Connecting Dataset->Model_Wrapper

TODO

## Training

TODO

## Prediction

TODO
