# Tagging System of Questions using Transfer Learning

## Overview

In this challenge, we provide the titles, text, and tags of Stack Exchange questions from six different sites. We then ask for tag predictions on unseen physics questions. Solving this problem via a standard machine learning approach might involve training an algorithm on a corpus of related text. Here, you are challenged to train on material from outside the field. Can an algorithm predict appropriate physics tags after learning from biology, chemistry or mathematics data? Let's find out!


## Objective

Main goal of this task is to train a model on questions belonging to domains like biology, chemistry, or mathematics but use that to predict tags of physics question.  These tags describe the topic of questions. 

## Dataset

In this dataset, you are provided with question titles, content, and tags for Stack Exchange sites on a variety of topics (biology, cooking, cryptography, diy, robotics, and travel). The content of each question is given as HTML. The tags are words or phrases that describe the topic of the question. The test set is comprised of questions from the [physics.stackexchange.com](https://physics.stackexchange.com)


## Approach

We use a deep learning based approach to tackle the problem. This method is inspired from the works of a submission made by Students, of stanford the relevant papar can be found in the repository.

 we convert the predicting of tags to a binary classiﬁcation problem. For each post (”title” and ”content”), we go through every tags and predict whether this tag is true or false. Then, we train a binary RNN model to do the binary classiﬁcation. This binary RNN model takes title (or ”title + content”) as input and append the true tags to the end of the input. In the case of tags such as ”molecular-dynamics” we append both of them (”molecular”, ”dynamics”) at the end of the input. Then, we give each post a ”0/1” variable as a response indicating whether the tag is associated with the title. By training this binary RNN model, we can predict on the physics tags by running through each possible physics tag on each physics title (or”title+content”) to see whether we should include the tag as an output.
 

