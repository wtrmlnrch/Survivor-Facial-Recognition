# HW1: Face Recognition

> "In the game of _Survivor_, every face tells a story, but only the ones who can read the unseen will find their way to victory."
>
> &mdash; <cite>ChatGPT</cite>

![alt text](https://github.com/FloridaSouthernCS/csc4510-f24-hw1/blob/main/data/survivor_montage.png "Survivor Castaways")

Face recognition is a classic real-world application for **_unsupervised learning_** techniques. In this assignment, you will work in pairs to develop Python code that leverages principal component analysis (PCA) and k-means clustering along with supervised learning methods such as nearest neighbors and neural networks to explore an image dataset related to the reality TV show _Survivor_.

## Data

The image data required for this problem has been provided for you. The `data/survivor` folder comprises 839 images (70x70 pixels each) of the castaways from the first 46 seasons of _Survivor_. The filenaming convention is `S##_first_last.png`, where ## refers to the season number; note that names may not always follow the exact structure of `first_last` due to middle names, multiple last names, etc. Also, the host of the show is included as `S00_Jeff_Probst.png`. All of the _Survivor_ data will be used for "training".

The `data/professors` folder comprises 5 images (70x70 pixels each) of the full-time faculty in the computer science department at Florida Southern College. The filenaming convention is `first_last.png`. This data will be used for "testing".

## Instructions

Starter code has been provided for you in `face_recognition.py`. Your programming tasks are as follows:

1. Apply **PCA** to the _Survivor_ faces dataset in order to reduce dimensionality while maintaining at least 90% of the original variance. You are encouraged to use the [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) methods in the scikit-learn library.

2. _Which professor looks least like a face according to the underlying facial features in the Survivor dataset?_ To answer this question, reconstruct each professor's face using the limited number of principal components from (1), then compute the Euclidean distance from the reconstructed face to the original. Largest distance indicates least likely to be a "face".

3. _Which professor is most likely to be the next host of Survivor?_ To answer this question, project each professor into the reduced "Survivor face space" and apply **nearest neighbor** classification to see who looks most similar to Jeff Probst.

4. _Which season would each professor most likely be on?_ To answer this question, use **k-means clustering** on the PCA-reduced _Survivor_ faces, then assign each of the PCA-reduced professor faces to the nearest cluster. The average season of _Survivor_ castaways in the cluster (not including Jeff Probst) is the assigned season for that professor.

5. _Which professor is most likely to win Survivor?_ Be creative! You must justify your answer to this question in a quantitative way using the results of PCA on the _Survivor_ dataset.

**_NOTE: For all tasks outlined above, use appropriate display techniques in your code to show output to the user (matplotlib and scikit-image may be useful libraries)._**

In addition to code, you must create a short PDF that **_briefy_** discusses the results of your program. Use images or plots as needed to convey the appropriate information. Add the PDF to your repo prior to submission.

## Submission Requirements

To earn credit for this assignment, you must commit and push any code changes to your group repository by the posted deadline (September 25, 2024 @ 9:25 AM EST). When you are ready for the instructor to review your code, please send a direct message on Slack.

Recall from the course syllabus that students are strongly encouraged to submit assignments well in advance of the deadline because they will be allowed to edit and resubmit each homework assignment for grade improvement, if they desire. Assignments can be resubmitted multiple times, as long as the grade improves each time.
