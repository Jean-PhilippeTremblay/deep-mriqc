# deep-mriqc

A [Deep Brainhack 2017](https://brainhack101.github.io/deepbrainhack2017/) project.
---

### Motivation:
---
A deep-learning approach to [MRIQC](http://www.biorxiv.org/content/early/2017/08/22/111294). Using subjects from the publicly accessible [ABIDE dataset](http://fcon_1000.projects.nitrc.org/indi/abide/) and ratings from expert reviewer(s), we trained a 4-layer convolutional neural network (CNN) to classify raw MRI images as high- or low-quality.

### Requirements:
---
The code is implemented in python3, using a [keras](https://keras.io/) wrapper on [tensorflow](https://www.tensorflow.org/). All code was tested on GPUs generously provided by [ElementAI](https://www.elementai.com/).

### Next Steps:
---
At the hackathon, our highest prediction accuracy was 78%. However, this was for high-confidence (i.e., "accept" or "exclude") ratings from a single expert reviewer (rater2) when pooling across all 17 ABIDE sites. Future directions include allowing for low-confidence ratings (i.e., "doubtful"), combining ratings across reviewers, and assessing performance for individual sites.

### Questions?
---
Open an issue! Or a pull request, if you're feeling particularly nice.
