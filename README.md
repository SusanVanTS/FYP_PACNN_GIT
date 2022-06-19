## Intro
Crowd Estimation and Understanding is an important study to maintain public safety and execute public control. Too dense in an area will cause many deaths and injuries when stampede happened. All of these accidents can be avoided when people are detected to be too crowded in somewhere. In this study, I have used Density Peak Clustering to assist me in understanding the crowds. 

PS: PACNN (Perspective Awareness Convolutional Neural Network)

## Objective
+ To identify and list the coordinates of the denser areas in the crowd. 
+ Modify the way of calculating the distance between 2 points by implicating perspective values
 
## Dataset
+ The perspective values from the PACNN output 
+ The ground truth values from the ShanghaiTech dataset 

## Process
“Mosh-pit” dense area in crowd will be detected through a density-peak clustering algorithm by calculating the space between crowds. The output of this project is the estimated crowd and the maximum density of that crowd. A list of coordinates and mosh-pits dense area will be generated and images of crowds marked with mosh-pits dense will be generated. 

![](/Flow%20Chart.png)

## Limitation
The limitation for this project is that it cannot detect real -time conditions because it analyzes images, not videos. 

## Obstruction
The problem for this project is that the images needed for this model must be in high resolution. 

## Further Suggestion
In order to make this the model can be used to implement the actual system:
- Complete this project is to use PACNN in the system in order to be able to analyze the image directly, not get data from ground truth value manually. 
- Implement the algorithm in video/live tracking system, not in images

## Last Word
During an event, the control room can use this model to monitor if people are too crowded. This is to avoid a human stampede to happen. 
