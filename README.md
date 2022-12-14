# Mars Image Classification 


<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/Cover.jpg"/>

## Introduction

The exploration of the planet Mars is a race for the space sovereignty of most developed countries. It is part of a huge geopolitical and scientific adventure. Our interest in this planet is based on the fact that there is evidence that Mars was home to life and that it is possible to go and live there one day. However, before embarking on such a journey, it is necessary to study it and learn its secrets. The Mars Reconnaissance Orbiter is a satellite that orbits the planet to take pictures of its soil and study its meteorology. The data in the form of images that we have of Mars are of considerable size, hence the importance of inferring knowledge from them. The images taken of Mars are scattered, so a classification of them could well help scientists to better understand the Martian soil in a general way and more particularly in terms of geological history, physico-chemical composition and the choice of relevant trajectories for the rovers on the Martian soil. 

This project is thus the subject of a classification of images of Mars. After a first introductory part, this report is subdivided in four main chapters and closed by a conclusion. The first chapter presents the framework of the Mars mission and outlines its scientific objectives. The second chapter presents the dataset on which we will work and solves some imbalance problems. The third chapter designs the architecture of the convolutional neural network to be used as well as the methods for optimizing the hyperparameters of the model and for evaluating its performance. Finally, the last chapter synthesizes and comments the obtained results.

## Description of the work environment 

### About the mission

On August 12, 2005, the U.S. space agency NASA launched a multi-purpose spacecraft called the Mars Reconnaissance Orbiter, which aims to advance our understanding of the red planet "Mars" through meticulous observation of its environment. These observations will examine potential landing sites for future surface missions and provide a high-speed communications relay for these missions.

The orbiter will be used to study the Martian surface, monitor the atmosphere, and probe the subsurface to learn more about the history and distribution of water on Mars in all its forms: frozen, liquid, or vapor. With the most powerful telescopic camera ever sent to another planet, the Mars Reconnaissance Orbiter will be able to show features of the Martian landscape at very small scales from the spacecraft's low altitude. This camera, along with five other science instruments, will produce numerous data each day during the planned 24-month science phase that will be sent back to Earth at a rate more than 10 times faster than previous Mars missions. The spacecraft will use a larger dish antenna, a faster computer, and an amplifier powered by a larger solar cell array.

### Technical description

Mars Reconnaissance Orbiter is 6.5 meters high and has a 3 meters high parabolic antenna. The solar panels are made of 20 square meters of solar cells. The spacecraft weighed 2,180 kilograms at launch, and the fuel for the 20 onboard thrusters was about half that mass.

In order to process huge amounts of data, the orbiter has 160 gigabits of solid-state memory and a processor with 46 million instructions per second.

### Science Objectives

The science objectives of the Mars Reconnaissance Orbiter mission will serve as a critical step in NASA's Mars exploration program in terms of its water and bacterial life resources and its climatic, geological, and environmental changes; in other words, in terms of its potential to harbor life.

The key objectives of this mission are:

	- To understand the Martian climate and its seasonality.
	- Characterize the Martian atmosphere and monitor its meteorology.
	- Study complex terrain and identify water-related landforms.
	- Search for sites with stratigraphic or compositional evidence of water or hydrothermal activity.
	- Search the subsurface for evidence of subterranean layers, water and ice, and profile the internal structure of polar ice caps.
	- Identify and characterize sites with the greatest potential for future surface missions.
	- Communicate scientific information from Mars surface missions back to Earth.
 
 ### Instruments on board

The orbiter carries six scientific instruments to examine the planet in various parts of the electromagnetic spectrum (from ultraviolet to radio waves).

Camera HiRISE : (High Resolution Imaging Science Experiment) consists of a telescope of 50 cm diameter and a focal length of 12 m. Its spatial resolution on the ground reaches 30 cm from an altitude of 300 km. The detector consists of 14 CCD sensors of 2 048 x 128 pixels, 10 of which have a red filter and the other four a blue, green and near infrared filter. It takes pictures in three color bands: blue-green, red and infrared. To facilitate the mapping of potential landing sites, the HiRISE camera can produce stereo images.

### Formulation of the problem

The HiRISE camera is one of the largest and most precise ever embarked in an orbital probe, it can acquire up to 28 gigabits of data in less than 6 seconds. This huge amount of images must be processed so that we can make the most of them and meet our objectives of detecting possible areas of water on Mars and defining sites and trajectories with high potential for surface missions.

Our goal is therefore to develop a machine learning model to classify a large database of images taken by HiRISE annotated by NASA geologist and physicist experts for deployment on one of the orbiter components publicly available on the website www.zenodo.org and published on September 16, 2020 by scientists from NASA's Jet Propulsion Laboratory.

Among the benefits of this operation is the automation of future processing to detect areas of interest on the Martian soil and the minimization of sending information of low inferential quality to Earth.

## Dataset description
Our dataset consists of a collection of 10815 black and white images of size 227x227 pixels taken by the HiRISE camera of the Mars Reconnaissance Orbiter. It is composed of 8 classes annotated by scientists from NASA's Jet Propulsion Laboratory. The table below summarizes the information of each class and gives their meanings.


<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/classes.PNG"/>

### Gallery 

<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/Gallery.PNG"/>

 
## Data Preprocessing

From the table of counts and figures below, we can conclude that our dataset is obviously unbalanced. There is a large presence of samples from class 0 (81.39% of the data) and class 1 (7.34% of the data) against a tiny fraction of samples for class 5 (0.68% of the data). We observe that the majority of the classes have counts that wobble around 300 samples, so we will bring all the classes down to this number by resampling.

<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/Percent.PNG"/>

### Oversampling
To reach the 300 samples, classes 2, 3, 4, 5, 6 and 7 need more samples than they have, so we will oversample them.

To achieve such a result, we opted for the SMOTE (Synthetic Minority Over-sampling Technique) method, which generates synthetic new samples from linear combinations of the original samples. This choice was based on its computational speed as well as its great advantage of generating new images, something that will allow our neural network, to be trained later, to better detect the characteristics of images of this class

### Subsampling

For the two classes 0 and 1, we performed clustering by the K-Means algorithm on the vector flattened raster images with the number of clusters 300 chosen in such a way to balance the dataset. After clustering, we take one representative from each group (closest to the centroid) and form our class from this new dataset. The clustering was chosen in order to keep the information homogeneous with respect to the whole dataset and not to have images with features unique to the class in question.

<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/distribution.PNG"/>

We observe a nice uniformity in the distribution of class counts, and in view of the methods chosen to obtain such a result, a large part of the information of the whole dataset is kept.

## Empirical results

This section shows the results of our simulations before and after balancing our database. The results after balancing are divided into both a simulation with consideration of the "other" class and one without. In all simulations, the initial parameters are:

	- Training percentage: 70% of the data
	- Validation percentage: 15% of the data
	- Percentage of testing: 15% of the data
	- Number of trials for the HyperBand algorithm: 30
	- Number of iterations per trial: 3
	- Number of epochs: 30

Using the new balanced distributions of the image classes obtained by under- and oversampling, we can now and oversampling, we can now measure globally the performance of our models the performance of our models thanks to the famous accuracy score.

### With the "other" class
We notice that the distributions of the training sets are better balanced compared to the previous example.

<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/with_class0.PNG"/>

We note the very poor accuracy value with 33% of good classifications. The two classes 5 and 6 are totally sucked into the other classes. No f1-score value admits more than 50% of good classifications.

The above results make us wonder about the problem that could occur even after a good balancing of the data that takes into account the statistical information of the data set.

### Without the "other" class

After some thought, we were able to discover that the "other" class is the source of our problem. It is a class that, by definition, includes images that do not fit into any of the above categories. It is therefore a collection of heterogeneous images for which learning features is most difficult, because it is quite likely that no distinctive feature exists.

This reflection led us to try to remove the "other" class that does not add information to our learned classification and to consider only the remaining well annotated classes.

<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/without_class0.png"/>

####  Classification report after balancing without class 0 :

<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/Classification_report.PNG"/>

We find 86% of accuracy with f1-scores per class higher than 70%. We can then affirm that our classification, in spite of its rather heavy structure in convolutional layers necessary for a good extraction of the characteristics, is of good quality considering the scores and having the "other" class omitted.

### Synthesis
As a synthesis of the results obtained both on the architectural level of the neural network and on the performance level, we can conclude the following:

	- The "other" class is a heterogeneous class, something that affected the ability of the network to find features of its elements.
	- Despite the balancing of the data, the "other" class made the complexity of the model very high and the accuracy poor.
	- We notice that as long as there are homogeneous class data, the convolution layers increase to detect the finest features of each class.
	- The last optimal architecture responds to an empirical datum in the literature on the correct definition of a model stating the smooth and non-sudden decay of the number of parameters at the layers of the multi-layer perceptron.




