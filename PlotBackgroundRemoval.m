clear; clc; close all;

folderOriginal = "Banana Ripeness Classification Dataset/train";
kelasFolder = {'unripe', 'ripe', 'rotten', 'overripe'};
folderOutput = 'Clean_Banana';

%Total gambar yang dipakai sebesar 1000
maximumTotalGambar = 1000;

%Setiap kelas memiliki jumlah total gambar yang sama
maximumGambarPerKelas = maximumTotalGambar/4;

% Menghapus dan membuat folder kembali
if exist(folderOutput, 'dir')
    rmdir(folderOutput, 's');
end
mkdir(folderOutput);


disp("Memulai background removal...");

for i = 1:numel(kelasFolder)
    %Inisialisasi masing-masing path folder dan gambar
    kelas = kelasFolder{i};
    folderInput = fullfile(folderOriginal, kelas);     
    folderOutput = fullfile(folderOutput, kelas);
    mkdir(kelasFolderOutput);
    
    %Inisialisasi gambar dari folder yang diinput per kelas dan maximum
    %total
    folderGambar = dir(fullfile(folderInput, '*.jpg'));
    totalGambar = min(numel(folderGambar), maximumGambarPerKelas);
    folderGambar = folderGambar(1:totalGambar);

    for j = 1:length(folderGambar)
        pathFileGambar = folderGambar(j).name;
        fileGambar = fullfile(folderInput, pathFileGambar);
        gambarRgb = imread(fileGambar);

        %Setiap gambar memiliki masking yang berbeda, sehingga saling
        %memanggil fungsi masking masing-masing dan memasukan gambarnya
        switch kelas
            case 'ripe'
                [mask, ~] = RipeMask(gambarRgb);
            case 'rotten'
                [mask, ~] = RottenMask(gambarRgb);
            case 'unripe'
                [mask, ~] = UnrippedMask(gambarRgb);
            case 'overripe'
                [mask, ~] = OverripeMask(gambarRgb);
        end

        %Mengisi bagian lubang terbesar
        unfilled_mask = bwareafilt(mask, 1); 
        mask = imfill(unfilled_mask, 'holes');

        %Mengapply masknya masing-masing dan mengaruhnya di output_folder
        %"Clean_Banana"
        gambarMasked = bsxfun(@times, gambarRgb, cast(mask, 'like', gambarRgb));
        imwrite(gambarMasked, fullfile(folderOutput, pathFileGambar));
        disp("Image of " + kelas + ": " + j);
        
        %Plotting proses gambar awal, setelah di mask, dan hasil mask
        subplot(2, 2, 1);
        imshow(gambarRgb);
        title('Original Gambar RGB');
        
        subplot(2, 2, 3);
        imshow(unfilled_mask);
        title('Binary Mask Awal');
        
        subplot(2, 2, 4);
        imshow(mask);
        title('Binary Mask Setelah di Filled');
        
        subplot(2, 2, 2);
        imshow(gambarMasked);
        title('Masked Gambar RGB');
        
        %Supaya transition plot
        drawnow;          
        pause(0.5);  
    end
end
disp("Background removal selesai.");
