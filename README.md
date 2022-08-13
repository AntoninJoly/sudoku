# Sudoku solver using a combination of CV, DL and rule based algorithms
Computer vision & machine learning were considered early to solve this problem but DL was more performant, thus chosen. The whole implementation was tested on both notebook and RESTful api, so it could be deployed easily.
Interesting points of this hobby projects are:
- Grid detection using semantic segmentation
- Grid processing using computer vision (contouring, homography, grid & digit detection, grid clustering)
- Handwritten/editor digits classiication
- Problem solving
- Backtracking of solution in original point of view

Digit classification model was trained using a mix of MNIST data and editor-made data (auto-generated) using a simple CNN model. Evaluation of this model achieved 0.9714 average f1-score on test data.

Grid semantic segmentation was trained using handmade labelled data, which original images were taken from the net, using resnet50+unet like architecture. Evaluation of this model achieved IoU:0.9785 - Dice:0.9891 on test data.

# Overview & example 📈

![](/assets/example1.png)

# To be done next 🛠
Following steps are considered:
- Mobile app implementation
- Model speed up
- Grid detection hyperparameter optimization
