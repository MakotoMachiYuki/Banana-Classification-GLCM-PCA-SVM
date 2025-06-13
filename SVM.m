function SVM(DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika)
    % Hyperparameters
    sigma = 1.0;
    C = 1;
    maxIter = 100;
    lr = 0.1;

    % Label Kelas
    namaKelas = {'overripe', 'ripe', 'rotten', 'unripe'};


    %Training Accuracy tanpa PCA
    [model_no_PCA, listKelas] = SVM_Training(DataTrain_STD, labelTrain_Numerika, sigma, C, maxIter, lr);
    prediksi_training_no_PCA = SVM_Classification(model_no_PCA, DataTrain_STD, listKelas);
    akurasi_training_no_PCA = sum(prediksi_training_no_PCA == labelTrain_Numerika) / length(labelTrain_Numerika) * 100;
    disp(['Akurasi Training SVM Kustom tanpa PCA (RBF): ', num2str(akurasi_training_no_PCA, '%.2f'), '%']);
    
    %Test Accuracy (tanpa PCA)
    prediksi_testing_no_PCA = SVM_Classification(model_no_PCA, DataTest_STD, listKelas);
    akurasi_training_no_PCA = sum(prediksi_testing_no_PCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM Kustom tanpa PCA (RBF): ', num2str(akurasi_training_no_PCA, '%.2f'), '%']);
    
    %Training Accuracy dengan PCA
    [model_PCA, listKelas] = SVM_Training(PCA_TRAIN_HASIL, labelTrain_Numerika, sigma, C, maxIter, lr);
    prediksi_training_PCA = SVM_Classification(model_PCA, PCA_TRAIN_HASIL, listKelas);
    akurasi_prediksi_PCA = sum(prediksi_training_PCA == labelTrain_Numerika) / length(labelTrain_Numerika) * 100;
    disp(['Akurasi Training SVM Kustom dengan PCA (RBF): ', num2str(akurasi_prediksi_PCA, '%.2f'), '%']);
    
    %Test Accuracy dengan PCA
    prediksi_testing_PCA = SVM_Classification(model_PCA, PCA_TEST_HASIL, listKelas);
    akurasi_testing_PCA = sum(prediksi_testing_PCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM Kustom dengan PCA (RBF): ', num2str(akurasi_testing_PCA, '%.2f'), '%']);

    %Label untuk confusion matrix
    labelTest_Numerika = categorical(labelTest_Numerika, 1:numel(namaKelas), namaKelas);
    label_prediksi_no_PCA = categorical(prediksi_testing_no_PCA, 1:numel(namaKelas), namaKelas);
    label_prediksi_PCA   = categorical(prediksi_testing_PCA,   1:numel(namaKelas), namaKelas);
    
    %Confusion matrix
    figure('Name', 'Confusion Matrix - SVM Kustom tanpa PCA');
    confusionchart(labelTest_Numerika, label_prediksi_no_PCA, ...
        'Title', 'Custom SVM tanpa PCA', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    figure('Name', 'Confusion Matrix - SVM Kustom dengan PCA');
    confusionchart(labelTest_Numerika, label_prediksi_PCA, ...
        'Title', 'Custom SVM dengan PCA', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    figure('Name', 'Perbandingan Akurasi SVM Kustom');
    bar([akurasi_training_no_PCA, akurasi_prediksi_PCA; akurasi_training_no_PCA, akurasi_testing_PCA]);
    title('Perbandingan Akurasi Custom SVM');
    xlabel('Skenario');
    ylabel('Akurasi (%)');
    xticklabels({'Train', 'Test'});
    legend({'Tanpa PCA', 'Dengan PCA'}, 'Location', 'northwest');
    ylim([0 100]);
    grid on;
end

%% Model One Vs All SVM
function [model, listKelas] = SVM_Training(X, Y, sigma, C, maxIter, lr)
% X: Fitur dari training    
% Y: Numerikal Label
% sigma: Parameter Gaussian kernel
% C: Parameter Regulasi
% maxIter: Maksimum iterasi
% lr: Learning Rate

listKelas = unique(Y);
totalKelas = length(listKelas);
totalTrain = size(X, 1);

model = cell(totalKelas, 1);

%Precompute kernel matrix
K = Kernel_Gaussian(X, X, sigma);

for i = 1:totalKelas
    %Label binary untuk One-vs-All
    y_binary = double(Y == listKelas(i));
    y_binary(y_binary == 0) = -1;

    %Initialisasi alpha
    alpha = zeros(totalTrain, 1);

    % Gradient ascent untuk dual formulation
    for iter = 1:maxIter
        for j = 1:totalTrain
            f = sum(alpha .* y_binary .* K(:, j));
            alpha(j) = alpha(j) + lr * (1 - y_binary(j) * f);
            alpha(j) = min(max(alpha(j), 0), C);
        end
    end
    
    %Hitung bias
    supportIndex = find(alpha > 1e-4);
    b = mean(y_binary(supportIndex) - K(supportIndex, :) * (alpha .* y_binary));

    %Menyimpan model
    model{i} = struct('alpha', alpha, ...
                        'b', b, ...
                        'y', y_binary, ...
                        'X', X, ...
                        'sigma', sigma);
end

end

%% Klassifikasi SVM
function classification = SVM_Classification(models, Xtest, classlist)
% models: model yang dihasilkan dari SVM_Training
% XTest: fitur test
% ClassList: Label kelas

jumlahKelas = length(models);
jumlahTest = size(Xtest, 1);
scores = zeros(jumlahTest, jumlahKelas);

    for i = 1:jumlahKelas
        model = models{i};
        K_test = Kernel_Gaussian(Xtest, model.X, model.sigma);
        scores(:, i) = K_test * (model.alpha .* model.y) + model.b;
    end

    [~, maxIndex] = max(scores, [], 2);
    classification = classlist(maxIndex);
end


%% RBF Gaussian Kernel
function K = Kernel_Gaussian(X1, X2, sigma)
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    K = zeros(n1, n2);
    factor = 1 / (2 * sigma^2);

    for i = 1:n1
        for j = 1:n2
            diff = X1(i, :) - X2(j, :);
            K(i, j) = exp(-factor * (diff * diff'));
        end

    end
end



