Face recognition SVM
=================

Train multiple images per person then recognize known faces in an image using a SVC in Python

This program is based on ageitgey's [face_recognition](https://github.com/ageitgey/face_recognition) api for Python and [dlib](https://github.com/davisking/dlib). A support vector classifier (SVC) with scikit-learn is trained on the face encodings from all the known faces in the training directory. It then recognizes the faces found in a test_image. Please note that it will produce meaningless results on very small datasets.

Installation
------------

```bash
pip install -r requirements.txt
```

Usage
-----

```bash
Usage:
  face_recognize.py -d <train_dir> -i <test_image>

Options:
  -h, --help                Show this help
  -d, --train_dir=<train_dir>         Directory with images for training
  -i, --test_image=<test_image>          Test image
```

Training directory structure
-----

```
<train_dir>/
    <person_1>/
        <person_1_face-1>.jpg
        <person_1_face-2>.jpg
        .
        .
        <person_1_face-n>.jpg
    <person_2>/
        <person_2_face-1>.jpg
        <person_2_face-2>.jpg
        .
        .
        <person_2_face-n>.jpg
    .
    .
    <person_n>/
        <person_n_face-1>.jpg
        <person_n_face-2>.jpg
        .
        .
        <person_n_face-n>.jpg
```
