function [DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika] = PCA(variance)

    % Read data
    tabelTrain = readtable("fitur_train.csv");
    tabelTest  = readtable("fitur_test.csv");

    % Extract features
    X_Train = table2array(tabelTrain(:, 2:10));
    X_Test  = table2array(tabelTest(:, 2:10));

    % === Standardize using training mean and std ===
    muTrain = mean(X_Train);
    stdTrain = std(X_Train);
    
    X_Train_Tengah = X_Train - muTrain;
    X_std_train = X_Train_Tengah ./ stdTrain;

    X_Test_Tengah = X_Test - muTrain;
    X_std_test = X_Test_Tengah ./ stdTrain;

    DataTrain_STD = X_std_train;
    DataTest_STD  = X_std_test;

    % === PCA from training set only ===
    C = cov(X_std_train);
    [V, D] = eig(C);

    [EigenValues, index] = sort(diag(D), 'descend');
    V_sorted = V(:, index);

    Total = sum(EigenValues);
    proporsiValue = EigenValues / Total;

    PK = 0;
    komponen = max(0, 2);  % start with at least 2 components
    for j = 1:length(proporsiValue)
        if PK < variance
            PK = PK + proporsiValue(j);
            komponen = komponen + 1;
        else
            break;
        end
    end

    disp(['Variabel atau komponen yang diperlukan untuk TRAIN ', num2str(variance * 100, "%.2f"), ...
        '% variance: ', num2str(komponen)]);

    % Project both datasets
    PCA_TRAIN_HASIL = X_std_train * V_sorted(:, 1:komponen);
    PCA_TEST_HASIL  = X_std_test  * V_sorted(:, 1:komponen);

    % === Label encoding ===
    labelTrain_Kelas = tabelTrain.Kelas;
    labelTrain_Numerika = grp2idx(labelTrain_Kelas);
    labelTrain_KelasUnik = unique(labelTrain_Kelas);

    labelTest_Kelas = tabelTest.Kelas;
    labelTest_Numerika = grp2idx(labelTest_Kelas);
    labelTest_KelasUnik = unique(labelTest_Kelas);

    warnaKelas = [
        43 43 12;   % overripe (hitam)
        255 255 0;  % ripe (kuning)
        255 0 0;    % rotten (merah)
        0 255 0     % unripe (hijau)
    ] / 255;

    % === PCA 2D Plot ===
    figure('Name', 'PCA 2D', 'NumberTitle', 'off'); hold on;
    for i = 1:numel(labelTrain_KelasUnik)
        index = strcmp(labelTrain_Kelas, labelTrain_KelasUnik{i});
        scatter(PCA_TRAIN_HASIL(index,1), PCA_TRAIN_HASIL(index,2), ...
            10, warnaKelas(i,:), 'filled', 'DisplayName', labelTrain_KelasUnik{i});
    end
    title('PCA 2D');
    xlabel('PC 1'); ylabel('PC 2');
    grid on; axis equal;
    legend('Location', 'best');

    % === PCA 3D Plot (if >= 3 components) ===
    if komponen >= 3
        figure('Name', 'PCA 3D', 'NumberTitle', 'off'); hold on;
        for i = 1:numel(labelTrain_KelasUnik)
            index = strcmp(labelTrain_Kelas, labelTrain_KelasUnik{i});
            scatter3(PCA_TRAIN_HASIL(index,1), PCA_TRAIN_HASIL(index,2), PCA_TRAIN_HASIL(index,3), ...
                10, warnaKelas(i,:), 'filled', 'DisplayName', labelTrain_KelasUnik{i});
        end
        title('PCA 3D');
        xlabel('PC 1'); ylabel('PC 2'); zlabel('PC 3');
        grid on; axis equal; view(45, 25);
        legend('Location', 'best');
    end
end
