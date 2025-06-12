function SVM(DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika)
    % Hyperparameters
    sigma = 1.0;
    C = 0.1;
    maxIter = 100;
    lr = 0.01;

    % Label Kelas
    classNames = {'overripe', 'ripe', 'rotten', 'unripe'};


    %Training Accuracy tanpa PCA
    [models_noPCA, classList] = SVM_Training(DataTrain_STD, labelTrain_Numerika, sigma, C, maxIter, lr);
    predTrain_noPCA = SVM_Classification(models_noPCA, DataTrain_STD, classList);
    accTrain_noPCA = sum(predTrain_noPCA == labelTrain_Numerika) / length(labelTrain_Numerika) * 100;
    disp(['Akurasi Training SVM Kustom tanpa PCA (RBF): ', num2str(accTrain_noPCA, '%.2f'), '%']);
    
    %Test Accuracy (tanpa PCA)
    predTest_noPCA = SVM_Classification(models_noPCA, DataTest_STD, classList);
    accTest_noPCA = sum(predTest_noPCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM Kustom tanpa PCA (RBF): ', num2str(accTest_noPCA, '%.2f'), '%']);
    
    %Training Accuracy dengan PCA
    [models_PCA, classList] = SVM_Training(PCA_TRAIN_HASIL, labelTrain_Numerika, sigma, C, maxIter, lr);
    predTrain_PCA = SVM_Classification(models_PCA, PCA_TRAIN_HASIL, classList);
    accTrain_PCA = sum(predTrain_PCA == labelTrain_Numerika) / length(labelTrain_Numerika) * 100;
    disp(['Akurasi Training SVM Kustom dengan PCA (RBF): ', num2str(accTrain_PCA, '%.2f'), '%']);
    
    %Test Accuracy dengan PCA
    predTest_PCA = SVM_Classification(models_PCA, PCA_TEST_HASIL, classList);
    accTest_PCA = sum(predTest_PCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM Kustom dengan PCA (RBF): ', num2str(accTest_PCA, '%.2f'), '%']);

    %Label untuk confusion matrix
    labelTest_Numerika = categorical(labelTest_Numerika, 1:numel(classNames), classNames);
    ypred_noPCA = categorical(predTest_noPCA, 1:numel(classNames), classNames);
    ypred_PCA   = categorical(predTest_PCA,   1:numel(classNames), classNames);
    
    %Confusion matrix
    figure('Name', 'Confusion Matrix - SVM Kustom tanpa PCA');
    confusionchart(labelTest_Numerika, ypred_noPCA, ...
        'Title', 'Custom SVM tanpa PCA', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    figure('Name', 'Confusion Matrix - SVM Kustom dengan PCA');
    confusionchart(labelTest_Numerika, ypred_PCA, ...
        'Title', 'Custom SVM dengan PCA', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    figure('Name', 'Perbandingan Akurasi SVM Kustom');
    bar([accTrain_noPCA, accTrain_PCA; accTest_noPCA, accTest_PCA]);
    title('Perbandingan Akurasi SVM');
    xlabel('Skenario');
    ylabel('Akurasi (%)');
    xticklabels({'Train', 'Test'});
    legend({'Tanpa PCA', 'Dengan PCA'}, 'Location', 'northwest');
    ylim([0 100]);
    grid on;
end

%% Model One Vs All SVM
function [models, classList] = SVM_Training(X, Y, sigma, C, maxIter, lr)
% X: Fitur dari training    
% Y: Numerikal Label
% sigma: Parameter Gaussian kernel
% C: Parameter Regulasi
% maxIter: Maksimum iterasi
% lr: Learning Rate

classList = unique(Y);
numClasses = length(classList);
numTrain = size(X, 1);

models = cell(numClasses, 1);

%Precompute kernel matrix
K = Kernel_Gaussian(X, X, sigma);

for i = 1:numClasses
    %Label binary untuk One-vs-All
    y_binary = double(Y == classList(i));
    y_binary(y_binary == 0) = -1;

    %Initialisasi alpha
    alpha = zeros(numTrain, 1);

    % Gradient ascent untuk dual formulation
    for iter = 1:maxIter
        for j = 1:numTrain
            f = sum(alpha .* y_binary .* K(:, j));
            alpha(j) = alpha(j) + lr * (1 - y_binary(j) * f);
            alpha(j) = min(max(alpha(j), 0), C);
        end
    end
    
    %Hitung bias
    supportIndex = find(alpha > 1e-4);
    b = mean(y_binary(supportIndex) - K(supportIndex, :) * (alpha .* y_binary));

    %Menyimpan model
    models{i} = struct('alpha', alpha, ...
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

numClasses = length(models);
numTest = size(Xtest, 1);
scores = zeros(numTest, numClasses);

    for i = 1:numClasses
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



