function [DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika] = PCA(variance, splitMargin)

    tabelTrain = readtable("fitur_train.csv");
    tabelTest = readtable("fitur_test.csv");


    %Menggunakan 4 fitur GLCM + 3 fitur HSV
    X_Train = table2array(tabelTrain(:, 2:8));
    X_Test  = table2array(tabelTest(:, 2:8));

    [n_train, p] = size(X_Train);
    [n_test, ~]  = size(X_Test);
    
    %Train Data
    %Mencari titik tengah dengan mengurangi mean
    meanTrain = mean(X_Train);
    Z_train = zeros(n_train, p);
    for i = 1:n_train
        Z_train(i,:) = X_Train(i,:) - meanTrain;
    end

    %Standardisasi data
    S = cov(Z_train);  
    Zr_train = zeros(n_train, p);
    for i = 1:n_train
        for j = 1:p
            Zr_train(i,j) = Z_train(i,j) / sqrt(S(j,j)); 
        end
    end
    DataTrain_STD = Zr_train;

    Zcov = cov(Zr_train);

    %Mendapatkan nilai eigenvalue dan eigenvector
    [V, D] = eig(Zcov);

    %Mengurutkan nilai eigenvalue dan eigenvector sekara descending
    m = 1;
    for i = p:-1:1
        DA_urut(m,m) = D(i,i);
        VA_urut(:,m) = V(:,i);
        m = m+1;
    end

    for i = 1:p
        eValue(i,1) = DA_urut(i,i);
    end

    Total = sum(eValue);
    PK = 0;
    komponen = 0;
    for j = 1:p
        proporsiValue(j,1) = eValue(j,1) / Total;
        if PK < variance
            PK = PK + proporsiValue(j,1);
            komponen = komponen + 1;
        else
            break;
        end
    end

    disp(['Variabel / Komponen untuk TRAIN ', num2str(variance * 100, '%.2f'), '%: ', num2str(komponen)]);
    
    VA_Komponen = VA_urut(:, 1:komponen);

    %Menhitung hasil PCA untuk data train
    PCA_TRAIN_HASIL = Zr_train * VA_Komponen;

    %Test Data
    %Mencari titik tengah
    Z_test = zeros(n_test, p);
    for i = 1:n_test
        Z_test(i,:) = X_Test(i,:) - meanTrain;
    end

    %Standardisasi
    Zr_test = zeros(n_test, p);
    for i = 1:n_test
        for j = 1:p
            Zr_test(i,j) = Z_test(i,j) / sqrt(S(j,j));  
        end
    end
    DataTest_STD = Zr_test;

    %Menghitung hasil PCA test
    PCA_TEST_HASIL = Zr_test * VA_Komponen;

    %Inisialisasi variable untuk label plot
    labelTrain_Kelas = tabelTrain.Kelas;
    labelTest_Kelas  = tabelTest.Kelas;
    labelTrain_Numerika = grp2idx(labelTrain_Kelas);
    labelTest_Numerika  = grp2idx(labelTest_Kelas);

    %Pernamaan splitPersen dalam plotting
    splitPersen1 = splitMargin * 100;
    splitPersen2 = 100 - splitPersen1;
    titlePCA2D = [' PCA 2D: ', num2str(splitPersen1), '% Train / ', num2str(splitPersen2), '% Test'];
    titlePCA3D = ['PCA 3D: ', num2str(splitPersen1), '% Train / ', num2str(splitPersen2), '% Test'];

    warnaKelas = [
        43 43 12.;     % overripe (hitam)
        255 255 0.;    % ripe (kuning)
        255 0 0;       % rotten (merah)
        0 255 0;       % unripe (hijau)
    ] / 255;

    %PCA 2D plot untuk hasil PCA training
    labelTrain_KelasUnik = unique(labelTrain_Kelas);
    figure('Name', 'PCA 2D', 'NumberTitle', 'off');
    hold on;
    for i = 1:numel(labelTrain_KelasUnik)
        idx = strcmp(labelTrain_Kelas, labelTrain_KelasUnik{i});
        scatter(PCA_TRAIN_HASIL(idx,1), PCA_TRAIN_HASIL(idx,2), ...
            10, warnaKelas(i,:), 'filled', 'DisplayName', labelTrain_KelasUnik{i});
    end
    title(titlePCA2D);
    xlabel('PC 1'); ylabel('PC 2');
    grid on; axis equal; legend('Location', 'best');

    %PCA 3D Plot jika komponen ada 3 atau lebih
    if komponen >= 3
        figure('Name', 'PCA 3D', 'NumberTitle', 'off');
        hold on;
        for i = 1:numel(labelTrain_KelasUnik)
            idx = strcmp(labelTrain_Kelas, labelTrain_KelasUnik{i});
            scatter3(PCA_TRAIN_HASIL(idx,1), PCA_TRAIN_HASIL(idx,2), PCA_TRAIN_HASIL(idx,3), ...
                10, warnaKelas(i,:), 'filled', 'DisplayName', labelTrain_KelasUnik{i});
        end
        title(titlePCA3D);
        xlabel('PC 1'); ylabel('PC 2'); zlabel('PC 3');
        grid on; axis equal;
        view(45, 25);
        legend('Location', 'best');
    end
end
