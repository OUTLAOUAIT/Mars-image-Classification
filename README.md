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

<img src="https://github.com/OUTLAOUAIT/Mars-image-Classification/blob/main/Images/Gallery.PNG"/>

 
