function [hasil] = mainglcm(knn)
    D = 'D:\Danny\bas\dataTrain';
    D1 = 'D:\Danny\bas\dataUji'; %image dari android disimpan disini
    imagetrains =  dir(fullfile(D,'*.jpg'));
    nfiles=length(imagetrains);
    ciriLatih = [];
    kelasCiri = [];
    imagetest = dir(fullfile(D1,'gambar.jpg'));
    uji = imread(fullfile(D1,imagetest.name));
    %ubah K disini
    k = knn;
    pack
    %Loop Data Latih
    for ii=1:nfiles
        imagetrains(ii).name;
        gambarIni = fullfile(D,imagetrains(ii).name);
        ini_ajah = imread(gambarIni);
        grayImage = rgb2gray (ini_ajah);
        if ii>34
            kelasCiri = [kelasCiri; 0];
        else
            kelasCiri = [kelasCiri; 1];
        end

           glcm = graylevel_comat(grayImage);
    %       at = analisistekstur(glcm);
           ciriLatih = [ciriLatih; glcm];
    end
    train = fitcknn(ciriLatih,kelasCiri,'NumNeighbors',k,'Standardize',1);
    dataUji = rgb2gray(uji);
    glcmUji = graylevel_comat(dataUji);
    [label,score,cost] = predict(train,glcmUji)

    if label == 0
        hasil = 'HFMD'
    else
        hasil = 'Sehat'
    end
end
