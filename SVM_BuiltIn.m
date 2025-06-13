function SVM_BuiltIn(DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika)

    %Hyperparameters
    sigma = 1.0; 
    C = 1;
    maxIter = 100;
    
    %Template SVM dengan Gaussian (RBF) kernel
    t = templateSVM( ...
    'KernelFunction', 'rbf', ...
    'KernelScale', sigma, ...
    'BoxConstraint', C, ...
    'Standardize', true, ...
    'IterationLimit', maxIter ...
    );

    namaKelas = {'overripe', 'ripe', 'rotten', 'unripe'};

    %SVM tanpa PCA
    model_no_PCA = fitcecoc(DataTrain_STD, labelTrain_Numerika, 'Learners', t);
    CVSV_no_PCA = crossval(model_no_PCA, 'KFold', 5);
    loss_no_PCA = kfoldLoss(CVSV_no_PCA);
    akurasi_training_no_PCA = (1 - loss_no_PCA) * 100;
    disp(['Akurasi Training SVM_BuiltIn tanpa PCA (RBF): ', num2str(akurasi_training_no_PCA, '%.2f'), '%']);

    %SVM dengan PCA ---
    model_PCA = fitcecoc(PCA_TRAIN_HASIL, labelTrain_Numerika, 'Learners', t);
    CVSVM_PCA = crossval(model_PCA, 'KFold', 5);
    loss_PCA = kfoldLoss(CVSVM_PCA);
    akurasi_training_no_PCA = (1 - loss_PCA) * 100;
    disp(['Akurasi Training SVM_BuiltIn dengan PCA (RBF): ', num2str(akurasi_training_no_PCA, '%.2f'), '%']);

    
    %Testing tanpa PCA
    prediksi_testing_no_PCA = predict(model_no_PCA, DataTest_STD);   
    akurasi_testing_no_PCA = sum(prediksi_testing_no_PCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM BuiltIn tanpa PCA (RBF): ', num2str(akurasi_testing_no_PCA, '%.2f'), '%']);
    
 %   predictedLabelsText = classNames(ypred_noPCA);
 %   disp(table(predictedLabelsText(:), classNames(labelTest_Numerika(:)), ...
 %   'VariableNames', {'Predicted', 'Actual'}));
   
   
    %Testing dengan PCA
    prediksi_testing_PCA = predict(model_PCA, PCA_TEST_HASIL);
    akurasi_testing_PCA = sum(prediksi_testing_PCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM BuiltIn dengan PCA (RBF): ', num2str(akurasi_testing_PCA, '%.2f'), '%']);
    
    
    %Label untuk confusion matrix
    labelTest_Numerika = categorical(labelTest_Numerika, 1:numel(namaKelas), namaKelas);
    label_prediksi_no_PCA = categorical(prediksi_testing_no_PCA, 1:numel(namaKelas), namaKelas);
    label_prediksi_PCA   = categorical(prediksi_testing_PCA,   1:numel(namaKelas), namaKelas);

    %Confusion Matrix
    figure('Name', 'Confusion Matrix - SVM_BuiltIn tanpa PCA');
    confusionchart(labelTest_Numerika, label_prediksi_no_PCA, ...
        'Title', 'SVM BuiltIn tanpa PCA (RBF)', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    

    figure('Name', 'Confusion Matrix - SVM_BuiltIn dengan PCA');
    confusionchart(labelTest_Numerika, label_prediksi_PCA, ...
        'Title', 'SVM BuiltIn dengan PCA (RBF)', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    %Bar Chart Akurasi
    figure('Name', 'Perbandingan Akurasi SVM_BuiltIn');
    bar([akurasi_training_no_PCA, akurasi_training_no_PCA; akurasi_testing_no_PCA, akurasi_testing_PCA]);
    title('Perbandingan Akurasi SVM BuiltIn');
    xlabel('Skenario');
    ylabel('Akurasi (%)');
    xticklabels({'Train', 'Test'});
    legend({'Tanpa PCA', 'Dengan PCA'}, 'Location', 'northwest');
    ylim([0 100]);
    grid on;
end
