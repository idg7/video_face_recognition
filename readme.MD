# Code for training and comapring a face recognition based on videos

This was adapted, and stemmed from the repository [PyTorch-Computer-Vision-Cookbook](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook)


`myutils.py` contains functions to train and test video recognition, including tests to ascertain if the model identifies motion.
`train.py` applies the training of a transformer on face recognition from videos.
`next_frame_prediction_train.py` contains functions to train the model on next frame prediction and test it (predicting the next frame's embedding from VGG16, using L2 loss).
`videos_embeddings_rdm.py` and `stills_embedings_rdm.py` are used to create an RDM using the approproiate models using image and video recognition.
