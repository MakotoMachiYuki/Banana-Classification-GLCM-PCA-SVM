%% GLCM
function GLCM(splitMargin, totalGambarPerKelas)

    kelasFolder = {'overripe', 'ripe', 'rotten', 'unripe'};
    folder = "Clean_Banana";
    folderProcessedTrain = 'Processed/train';
    folderProcessedTest = 'Processed/test';
    
    dataTrain = {};
    dataTest = {};
    
    %Menghapus folder dan membuat folder kembali
    if exist("Processed", 'dir')
        rmdir("Processed", 's');
    end
    mkdir(folderProcessedTrain);
    mkdir(folderProcessedTest);


    for i = 1:length(kelasFolder)
    
        %Inisialiasi path
        namaKelas = kelasFolder{i};
        folderInput = fullfile(folder, namaKelas);
        folderTrainOutput = fullfile(folderProcessedTrain, namaKelas);
        folderTestOutput = fullfile(folderProcessedTest, namaKelas);
        
        %Membuat folder kelasnya
        mkdir(folderTrainOutput);
        mkdir(folderTestOutput);

        %Inisialisasi gambar dari folder yang diinput per kelas dan maximum total
        folderGambar = dir(fullfile(folderInput, '*.jpg'));
        totalGambar = min(numel(folderGambar), totalGambarPerKelas);
        folderGambar = folderGambar(1:totalGambar); % Cap to max
        randIdx = randperm(totalGambar);
        
        %Membagi data berdasarkan split margin dan total gambar
        %Jika split margin = 0.6 = 60% gambar dari total masuk kedalam train folder
        %dan 40% masuk dalam test
        numTrain = round(splitMargin * totalGambar);
        jumlahTrain = randIdx(1:numTrain);
        jumlahTest = randIdx(numTrain+1:end);
        %TRAIN
        for j = jumlahTrain
            %Extrasi fitur GLCM, RGB dan HSV
            [row, namaGambar, fitur] = extractFeatures(folderGambar(j), folderInput);
            dataTrain(end+1, :) = [row, fitur, {namaKelas}];
            copyfile(fullfile(folderInput, namaGambar), fullfile(folderTrainOutput, namaGambar));
        end
        
        %TEST
        for j = jumlahTest
            %Extrasi fitur GLCM, RGB dan HSV
            [row, namaGambar, fitur] = extractFeatures(folderGambar(j), folderInput);
            dataTest(end+1, :) = [row, fitur, {namaKelas}];
            copyfile(fullfile(folderInput, namaGambar), fullfile(folderTestOutput, namaGambar));
        end
    end
        
    %Memasukan hasil extrasi kedalam file csv train
    tabelTrain = cell2table(dataTrain, 'VariableNames', ...
        {'Nama', 'Kontras', 'Korelasi', 'Energi', 'Homogenitas', ...
        'Red', 'Green', 'Blue', ...
        'Hue', 'Saturasi', 'Value', ...
        'Kelas'});
    
    if exist('fitur_train.csv', 'file')
        delete('fitur_train.csv');
        delete('fitur_train.mat');
    end
    
    writetable(tabelTrain, 'fitur_train.csv');
    save('fitur_train.mat', 'tabelTrain');
    
    
    tabelTest = cell2table(dataTest, 'VariableNames', ...
    {'Nama', 'Kontras', 'Korelasi', 'Energi', 'Homogenitas', ...
    'Red', 'Green', 'Blue', ...
    'Hue', 'Saturasi', 'Value', ['' ...
    'Kelas']});
    
    if exist('fitur_test.csv', 'file')
        delete('fitur_test.csv');
        delete('fitur_test.mat');
    end
    
    writetable(tabelTest, 'fitur_test.csv');
    save('fitur_test.mat', 'tabelTest');

end


%% Fungsi untuk memanggil algoritma GLCM untuk memasukan 
function [row, namaGambar, features] = extractFeatures(fileStruct, folder)

    namaGambar = fileStruct.name;
    pathGambar = fullfile(folder, namaGambar);
    gambar = imread(pathGambar);
    
    %Membuat image menjadi grayscale
    grayImg = rgb2gray(gambar);
    
    % 4 fitur dari GLCM
    stats = Algoritma_GLCM(grayImg, [0 1]);
    contrast = stats.Contrast;
    correlation = stats.Correlation;
    energy = stats.Energy;
    homogeneity = stats.Homogeneity;
    
    % Mean dari RGB
    Rmean = mean2(gambar(:,:,1));
    Gmean = mean2(gambar(:,:,2));
    Bmean = mean2(gambar(:,:,3));
    
    % Mean dari HSV
    hsvImg = rgb2hsv(gambar);
    Hmean = mean2(hsvImg(:,:,1));
    Smean = mean2(hsvImg(:,:,2));
    Vmean = mean2(hsvImg(:,:,3));
    
    row = {namaGambar};
    features = {contrast, correlation, energy, homogeneity, ...
                Rmean, Gmean, Bmean, ...
                Hmean, Smean, Vmean};
end


%% Algoritma GLCM
function fitur = Algoritma_GLCM(grayImg, offset)
    % Parameter
    %Level sebagai intensitas piksel dibagi menjadi beberapa kategori
    level = 8;
    grayImg = im2uint8(grayImg);  % Pastikan rentang 0–255
    kuantisasi = floor(double(grayImg) / (256 / level));  % Kuantisasi 0–7

    % Offset, sudut
    % 0° = [0, 1]
    % 90° = [-1, 0]
    % 180° = [0, -1]
    % 270° = [1, 0]

    dx = offset(2);
    dy = offset(1);

    % Inisialisasi GLCM
    P = zeros(level, level);
    [rows, cols] = size(kuantisasi);

    for i = 1:rows
        for j = 1:cols

            %offset dari 0° hingga 270°
            ni = i + dy;
            nj = j + dx;
            if ni >= 1 && ni <= rows && nj >= 1 && nj <= cols
                baris = kuantisasi(i, j) + 1;     
                kolom = kuantisasi(ni, nj) + 1;
                P(baris, kolom) = P(baris, kolom) + 1;
            end
        end
    end

    % Normalisasi
    P = P / sum(P(:));

    % Matriks indeks i dan j
    [i, j] = meshgrid(1:level, 1:level);
    i = i'; j = j';

    % Rata-rata dan standar deviasi
    mu_i = sum(sum(i .* P));
    mu_j = sum(sum(j .* P));
    sigma_i = sqrt(sum(sum((i - mu_i).^2 .* P)));
    sigma_j = sqrt(sum(sum((j - mu_j).^2 .* P)));

    % Ekstraksi fitur
    fitur = struct();
    fitur.Contrast = sum(sum((abs(i - j)).^2 .* P));
    fitur.Correlation = sum(sum((i - mu_i) .* (j - mu_j) .* P)) / (sigma_i * sigma_j + eps);
    fitur.Energy = sum(sum(P.^2));
    fitur.Homogeneity = sum(sum(P ./ (1 + abs(i - j))));
end
