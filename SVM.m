function SVM(DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika)

sigma = 1.0;
C = 1.0;
maxIter = 100;
lr = 0.01;

% Train
[models, classList] = SVM_Training(DataTrain_STD, labelTrain_Numerika, sigma, C, maxIter, lr);

% Klassifikasi
classification = SVM_Classification(models, DataTest_STD, classList);

% Evaluate
akurasi = sum(classification == labelTest_Numerika) / length(labelTest_Numerika) * 100;
disp(['Test Accuracy: ', num2str(akurasi, '%.2f'), '%']);

% Confusion matrix
figure;
confusionchart(labelTest_Numerika, classification, ...
    'Title', 'Confusion Matrix - Custom One-vs-All SVM', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
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

models = cell(numClasses, 1)

%Precompute kernel matrix
K = Kernel_Gaussian(X, X, sigma);

for i = 1:numClasses
    fprintf('Training Class %d vs all\n', classList(i));
    
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
        model = models{i}
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
    factor = 1 / (2 * sigma^2)

    for i = 1:n1
        for j = 1:n2
            diff = X1(i, :) - X2(j, :);
            K(i, j) = exp(-factor * (diff * diff'));
        end

    end
end



