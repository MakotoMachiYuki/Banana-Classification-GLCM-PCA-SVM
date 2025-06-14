clear; clc; close all;


% Inisialisasi variabel
%Untuk split data train/test
splitMargin = 0.8;

%Total gambar yang akan di background removal sebagai data utama
maximumTotalGambar = 1000;

%Total gambar per kelas supaya semua kelas memiliki jumlah data yang setara
maximumGambarPerKelas = maximumTotalGambar/4;

%Maksimal variansi untuk jumlah komponen di metode PCA
variance = 0.91;


% BACKGROUND REMOVAL
%Jika sudah remove, bisa di comment saja function background removalnya
disp("Background Removal...");
timeStart = tic;
%BackgroundRemoval(maximumGambarPerKelas);
timeEnd = toc(timeStart);

disp("Background removal selesai.");
disp("Waktu Background Removal: " + string(seconds(timeEnd)), 'mm:ss.SSS');

% GLCM
disp("GLCM...");
timeStart = tic;
GLCM(splitMargin, maximumGambarPerKelas);
timeEnd = toc(timeStart);

disp("GLCM Selesai");
disp("Waktu Extrasi GLCM: " + string(seconds(timeEnd)), 'mm:ss.SSS');

% PCA

disp("PCA...");
timeStart = tic;
[DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika] = PCA(variance, splitMargin);
timeEnd = toc(timeStart);

disp("PCA Selesai");
disp("Waktu Reduksi PCA: " + string(seconds(timeEnd)), 'mm:ss.SSS');

disp("SVM ...")
timeStart = tic;
SVM(DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika, splitMargin);
timeEnd = toc(timeStart);

disp("SVM Selesai");
disp("Waktu Total SVM: " + string(seconds(timeEnd)), 'mm:ss.SSS');

