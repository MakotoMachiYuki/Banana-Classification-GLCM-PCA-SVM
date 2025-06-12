function SVM_BuiltIn(DataTrain_STD, DataTest_STD, PCA_TRAIN_HASIL, PCA_TEST_HASIL, labelTrain_Numerika, labelTest_Numerika)
    % Template SVM dengan Gaussian (RBF) kernel
    t = templateSVM('KernelFunction', 'rbf', 'Standardize', true);

    classNames = {'overripe', 'ripe', 'rotten', 'unripe'};

    % --- SVM tanpa PCA ---
    SVMModel_noPCA = fitcecoc(DataTrain_STD, labelTrain_Numerika, 'Learners', t);
    CVSVM_noPCA = crossval(SVMModel_noPCA, 'KFold', 5);
    loss_noPCA = kfoldLoss(CVSVM_noPCA);
    accuracy_noPCA = (1 - loss_noPCA) * 100;
    disp(['Akurasi Training SVM_BuiltIn tanpa PCA (RBF): ', num2str(accuracy_noPCA, '%.2f'), '%']);

    % --- SVM dengan PCA ---
    SVMModel_PCA = fitcecoc(PCA_TRAIN_HASIL, labelTrain_Numerika, 'Learners', t);
    CVSVM_PCA = crossval(SVMModel_PCA, 'KFold', 5);
    loss_PCA = kfoldLoss(CVSVM_PCA);
    accuracy_PCA = (1 - loss_PCA) * 100;
    disp(['Akurasi Training SVM_BuiltIn dengan PCA (RBF): ', num2str(accuracy_PCA, '%.2f'), '%']);

    

    % --- Testing tanpa PCA ---
    ypred_noPCA = predict(SVMModel_noPCA, DataTest_STD);   
    akurasi_noPCA = sum(ypred_noPCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM_BuiltIn tanpa PCA (RBF): ', num2str(akurasi_noPCA, '%.2f'), '%']);
    
 %   predictedLabelsText = classNames(ypred_noPCA);
 %   disp(table(predictedLabelsText(:), classNames(labelTest_Numerika(:)), ...
 %   'VariableNames', {'Predicted', 'Actual'}));
   
   
    % --- Testing dengan PCA ---
    ypred_PCA = predict(SVMModel_PCA, PCA_TEST_HASIL);
    akurasi_PCA = sum(ypred_PCA == labelTest_Numerika) / length(labelTest_Numerika) * 100;
    disp(['Akurasi Test SVM_BuiltIn dengan PCA (RBF): ', num2str(akurasi_PCA, '%.2f'), '%']);

    labelTest_Numerika = categorical(labelTest_Numerika, 1:numel(classNames), classNames);
    ypred_noPCA = categorical(ypred_noPCA, 1:numel(classNames), classNames);
    ypred_PCA   = categorical(ypred_PCA,   1:numel(classNames), classNames);

    % --- Confusion Matrix ---
    figure('Name', 'Confusion Matrix - SVM tanpa PCA');
    confusionchart(labelTest_Numerika, ypred_noPCA, ...
        'Title', 'SVM tanpa PCA (RBF)', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    

    figure('Name', 'Confusion Matrix - SVM dengan PCA');
    confusionchart(labelTest_Numerika, ypred_PCA, ...
        'Title', 'SVM dengan PCA (RBF)', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    % --- Bar Chart Akurasi ---
    figure('Name', 'Perbandingan Akurasi SVM');
    bar([accuracy_noPCA, accuracy_PCA; akurasi_noPCA, akurasi_PCA]);
    title('Perbandingan Akurasi SVM');
    xlabel('Skenario');
    ylabel('Akurasi (%)');
    xticklabels({'Train', 'Test'});
    legend({'Tanpa PCA', 'Dengan PCA'}, 'Location', 'northwest');
    ylim([0 100]);
    grid on;
end
