%% PCA
function [DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika] = PCA(variance)
%clear; clc;
%close all;

   tabelTrain = readtable("fitur_train.csv");
   tabelTest = readtable("fitur_test.csv");
    
   %GLCM + RGB + HSV (10 variabel)
   X_Train = table2array(tabelTrain(:, 2:10));
   X_Test = table2array(tabelTest(:, 2:10));

    
   %Mencari titik tengah dengan mengurangi mean
   X_Train_Tengah = X_Train - mean(X_Train);
   X_Test_Tengah = X_Test - mean(X_Test);
    
    
   %Standardisasi Data Train
   stdDev = std(X_Train_Tengah); 
   [n, p] = size(X_Train_Tengah);
   X_std_train = zeros(n, p);
   for i = 1:n
       for j = 1:p
           X_std_train(i,j) = X_Train_Tengah(i,j) / stdDev(j);
       end
   end

   DataTrain_STD = X_std_train;
    
   %Standardisasi Data Test
   stdDev = std(X_Test_Tengah); 
   [n, p] = size(X_Test_Tengah);
   X_std_test = zeros(n, p);
   for i = 1:n
       for j = 1:p
           X_std_test(i,j) = X_Test_Tengah(i,j) / stdDev(j);
       end
   end
   DataTest_STD = X_std_test;

   %Covariance matrix
   C = cov(X_Train_Tengah);
    
   %Eigenvector dan eigenvalues
   [V, D] = eig(C);
    
   %Mengurutkan eigenvalues secara descending
   [EigenValues, index] = sort(diag(D), 'descend');
   V_sorted = V(:, index);
        
   Total = sum(EigenValues);
   proporsiValue = EigenValues / Total;   
    
   PK = 0; 
   komponen = 0;
   komponen = max(komponen, 2);
    
   for j = 1:length(proporsiValue)
       if PK < variance
           PK = PK + proporsiValue(j);
           komponen = komponen + 1;
       else
           break;
       end
   end
    
   disp(['Variabel atau komponen yang diperlukan untuk TRAIN',num2str(variance * 100, "%.2f"),  ...
       '% variance: ', ...
       num2str(komponen)]);

   
   %Menghitung hasil dari PCA untuk TRAIN
   PCA_TRAIN_HASIL = X_Train_Tengah * V_sorted(:, 1:komponen);


      %Covariance matrix
   C = cov(X_Test_Tengah);
    
   %Eigenvector dan eigenvalues
   [V, D] = eig(C);
    
   %Mengurutkan eigenvalues secara descending
   [EigenValues, index] = sort(diag(D), 'descend');
   V_sorted = V(:, index);
        
   Total = sum(EigenValues);
   proporsiValue = EigenValues / Total;   
    
   PK = 0; 
   komponen = 0;
   komponen = max(komponen, 2);
    
   for j = 1:length(proporsiValue)
       if PK < variance
           PK = PK + proporsiValue(j);
           komponen = komponen + 1;
       else
           break;
       end
   end
    
   disp(['Variabel atau komponen yang diperlukan untuk TEST',num2str(variance * 100, "%.2f"),  ...
       '% variance: ', ...
       num2str(komponen)]);

   
   %Menghitung hasil dari PCA untuk TEST
   PCA_TEST_HASIL = X_Test_Tengah * V_sorted(:, 1:komponen);

   % Extrasi label
   labelTrain_Kelas = tabelTrain.Kelas;
   labelTrain_Numerika = grp2idx(labelTrain_Kelas);
   labelTrain_KelasUnik = unique(labelTrain_Kelas);
   labelTrain_Warna = lines(numel(labelTrain_KelasUnik)); 

   labelTest_Kelas = tabelTest.Kelas;
   labelTest_Numerika = grp2idx(labelTest_Kelas);
   labelTest_KelasUnik = unique(labelTest_Kelas);
   labelTest_Warna = lines(numel(labelTest_KelasUnik)); 
    
   warnaKelas = [
        43 43 12.;   % overripe (hitam)
        255 255 0.;   % ripe (kuning)
        255 0 0;   % rotten (merah)
        0 255 0;   % unripe (hijau)
    ] / 255;
    
    % Membuat PCA 2D plot
   figure('Name', 'PCA 2D', 'NumberTitle', 'off');
   hold on;
   for i = 1:numel(labelTrain_KelasUnik)
       index = strcmp(labelTrain_Kelas, labelTrain_KelasUnik{i});        
       scatter(PCA_TRAIN_HASIL(index,1), PCA_TRAIN_HASIL(index,2), ...
           10, warnaKelas(i,:), 'filled', 'DisplayName', labelTrain_KelasUnik{i});
   end
   title('PCA 2D');
   xlabel('PC 1'); ylabel('PC 2');
   grid on; axis equal;
   legend('Location', 'best');
   
   %Membuat PCA 3D plot jika komponen lebih dari 3
   if(komponen >= 3)
       figure('Name', 'PCA 3D', 'NumberTitle', 'off');
       hold on;
       for i = 1:numel(labelTrain_KelasUnik)
           index = strcmp(labelTrain_Kelas, labelTrain_KelasUnik{i});
           scatter3(PCA_TRAIN_HASIL(index,1), PCA_TRAIN_HASIL(index,2), PCA_TRAIN_HASIL(index,3), ...
               10, warnaKelas(i,:), 'filled', 'DisplayName', labelTrain_KelasUnik{i});
       end
       title('PCA 3D');
       xlabel('PC 1'); ylabel('PC 2'); zlabel('PC 3');
       grid on; axis equal;
       view(45, 25);
       legend('Location', 'best');
   end
end