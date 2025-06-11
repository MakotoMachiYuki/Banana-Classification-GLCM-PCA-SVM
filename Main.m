clear; clc; close all;

splitMargin = 0.6;
maximumTotalGambar = 1000;
maximumGambarPerKelas = maximumTotalGambar/4;
variance = 0.91;


% BACKGROUND REMOVAL
%Jika sudah remove, bisa di comment saja function background removalnya
timeStart = tic;
disp("Background Removal...");
BackgroundRemoval(maximumGambarPerKelas);
timeEnd = toc(timeStart);
disp("Waktu Background Removal: " + string(seconds(timeEnd)), 'mm:ss.SSS');

% GLCM
timeStart = tic;
disp("GLCM...");
GLCM(splitMargin, maximumGambarPerKelas);
timeEnd = toc(timeStart);
disp("GLCM Selesai");
disp("Waktu Extrasi GLCM: " + string(seconds(timeEnd)), 'mm:ss.SSS');

% PCA
timeStart = tic;
disp("PCA...");
[DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika] = PCA(variance);
timeEnd = toc(timeStart);
disp("PCA Selesai");
disp("Waktu Reduksi PCA: " + string(seconds(timeEnd)), 'mm:ss.SSS');


% SVM
timeStart = tic;
disp("SVM...");
SVM(DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika);
timeEnd = toc(timeStart);
disp("SVM Selesai");
disp("Waktu SVM: " + string(seconds(timeEnd)), 'mm:ss.SSS');

