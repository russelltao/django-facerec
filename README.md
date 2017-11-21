# 基于django进行人脸识别
# 基于facenet(tensorflow/resnetv1)进行人脸识别
* export PYTHONPATH=face/facenet/src
# 基于mtcnn进行人脸检测




**编译opencv**:
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  -D WITH_TBB=ON -D WITH_EIGEN=ON -D WITH_OPENCL=ON -D WITH_CUDA=ON -D BUILD_opencv_gpu=ON -D BUILD_opencv_gpuarithm=ON -D BUILD_opencv_gpubgsegm=ON -D BUILD_opencv_gpucodec=ON -D BUILD_opencv_gpufeatures2d=ON -D BUILD_opencv_gpufilters=ON -D BUILD_opencv_gpuimgproc=ON -D BUILD_opencv_gpulegacy=ON -D BUILD_opencv_gpuoptflow=ON -D BUILD_opencv_gpustereo=ON -D BUILD_opencv_gpuwarping=ON ..

图片集：

| Tables | du -sh | classes | files|
| ------------- |:-------------:| -----:| -----:| 
| megaface  | 489G | 672057 |   |
| megdatasets | 139G | 546412 | 3440176 |
| casia | 5.4G | 10591 | 500452 |
| casdatasets | 21G | 10591 | 497549 |

# du -sh megaface
# ls -l megaface/MegafaceIdentities_VGG |grep "^d"|wc -l

**_1、训练_**
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PYTHONPATH=/home/taohui/code/facenet/src

`对齐`
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py /data/faceimg/CASIA-WebFace \
/data/datasets/CASIA-WebFace_182 --image_size 182 --margin 44 \
--random_order --gpu_memory_fraction 0.22 \
& done
python src/align/align_dataset.py /data/thface/CASIA-WebFace /data/datasets/casia/casia_maxpy_dlib_182 --image_size 182

python src/my_train_softmax.py --logs_base_dir /home/taohui/face/logs/ \
--models_base_dir /home/taohui/face/models/facenet/ \
--data_dir /data/datasets/casia_maxpy_mtcnnpy_182/ \
--image_size 160 --model_def models.inception_resnet_v1 --optimizer RMSPROP \
--learning_rate -1 --max_nrof_epochs 200 --batch_size 100 --epoch_size 1000 \
--keep_probability 0.8 --random_crop --random_flip \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 \
--lfw_dir /home/taohui/face/lfw_mtcnnpy_160 \
--pretrained_model /home/taohui/face/models/facenet/20170912-101532/model-20170912-101532.ckpt-200000

tensorboard --logdir=/home/taohui/face/logs/ --port 6006

megaface:
python src/align/my_align_mtcnn.py \
/data/faceimg/images \
/data/datasets/images_182 \
--image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.25


**2.lfw验证**
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py /data/faceimg/lfw /data/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 \
& done

**3、图片集注册并验证**
纯脸（非160*160）
python test/testcompare.py /home/taohui/face/models/facenet/20170912-101532 \
--logfolder /home/taohui/face/logs/mycomparetest/ --resize \
/home/taohui/face/datasets/images_182 


找脸并对齐
python test/testcompare.py /home/taohui/face/models/facenet/20170911-190042 \
/data/faceimg/images/ --detect_face --resize \
--logfolder /home/taohui/face/logs/mycomparetest/ 

**4、分类器**
python src/classifier.py TRAIN /home/taohui/face/datasets/images_182 \
/home/taohui/face/models/facenet/20170911-190042 \
/home/taohui/face/models/images_classifier.pkl \
--batch_size 1000 --min_nrof_images_per_class 7 --nrof_train_images_per_class 5 \
--use_split_dataset; \
python src/classifier.py CLASSIFY /home/taohui/face/datasets/images_182 \
/home/taohui/face/models/facenet/20170911-190042 \
/home/taohui/face/models/images_classifier.pkl \
--batch_size 1000 --min_nrof_images_per_class 7 --nrof_train_images_per_class 5 \
--use_split_dataset > aaa
