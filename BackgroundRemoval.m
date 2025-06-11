function BackgroundRemoval(maximumGambarPerKelas)
folderOriginal = "Banana Ripeness Classification Dataset/train";
kelasFolder = {'unripe', 'ripe', 'rotten', 'overripe'};
folderOutput = 'Clean_Banana';

%maximumTotalGambar = 1000;
%maximumGambarPerKelas = maximumTotalGambar/4;

% Menghapus dan membuat folder kembali
if exist(folderOutput, 'dir')
    rmdir(folderOutput, 's');
end
mkdir(folderOutput);

for i = 1:numel(kelasFolder)
    %Inisialisasi masing-masing path folder dan gambar
    kelas = kelasFolder{i};
    folderInput = fullfile(folderOriginal, kelas);
    kelasFolderOutput = fullfile(folderOutput, kelas);
    mkdir(kelasFolderOutput);
    
    %Inisialisasi gambar dari folder yang diinput per kelas dan maximum
    %total
    folderGambar = dir(fullfile(folderInput, '*.jpg'));
    totalGambar = min(numel(folderGambar), maximumGambarPerKelas);
    folderGambar = folderGambar(1:totalGambar);
    disp("Jumlah gambar " + kelas + ": " + totalGambar);
    
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
        imwrite(gambarMasked, fullfile(kelasFolderOutput, pathFileGambar));
        disp("Image of " + kelas + ": " + j);
    end
end

disp("Background removal selesai.");
