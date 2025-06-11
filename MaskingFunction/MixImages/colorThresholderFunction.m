clear; clc;

folder = "";
file = "banana_ripe.png"

img = imread(fullfile(append(folder,file)));

colorThresholder(img);