function hasil = mainSURF(knn)
    clc;
    D = 'D:\Danny\bas\dataTrain';
    D1 = 'D:\Danny\bas\dataUji'; %export image from android here
    %image={im2double(imresize(imread('tangan01.jpg'),[256 256]));};
    imagetrains =  dir(fullfile(D,'*.jpg'));
    nfiles=length(imagetrains);
    ciriLatih = [];
    kelasCiri = [];
    imagetest = dir(fullfile(D1,'gambar.jpg'));
    uji = imread(fullfile(D1,imagetest.name));
    %ubah K disini
    k = knn;
    %files={f.name}; 

    %=================
    %Train all files in dataTrain Folder
    for ii=1:nfiles 
        % Load images
        image=im2double(imresize(imread(fullfile(D,imagetrains(ii).name)),[256 256]));
        image=rgb2gray(image);
        I2=image; 

        % Get the Key Points
        Options.upright=true;
        Options.tresh=0.0001;
        SurfExtract = ExtractSURF(I2,Options);

        % Put the landmark descriptors in a matrix
        %D1 = reshape([Ipts1.descriptor],64,[]);
        FeatureVec = reshape([SurfExtract.descriptor],64,[]);
        klaster = kmeans(FeatureVec,3,'Distance','cityblock');
        %features = D2(:);
        panjang = length(klaster);
        features = reshape(klaster, [1 panjang]);
        ciriLatih = [ciriLatih; features];
        if ii>34
            kelasCiri = [kelasCiri; 0];
        else
            kelasCiri = [kelasCiri; 1];
        end
    end

    train = fitcknn(ciriLatih,kelasCiri,'NumNeighbors',k,'Standardize',1);
    %==========
    %Extract Test file from dataUji folder
    dataUji = im2double(imresize(uji,[256 256]));
    dataUji = rgb2gray(dataUji);
    % Get the Key Points
    Options.upright=true;
    Options.tresh=0.0001;
    SurfExtract = ExtractSURF(dataUji,Options);
    FeatureVec = reshape([SurfExtract.descriptor],64,[]);
    klaster = kmeans(FeatureVec,3,'Distance','cityblock');
    panjang = length(klaster);
    Testfeatures = reshape(klaster, [1 panjang]);
    % tempUji = detectSURFFeatures(dataUji);
    % [im_features, temp] = extractFeatures(dataUji, temp);
    % features = im_features(:);
    % panjang = length(features);
    [label,score,cost] = predict(train,Testfeatures);
    % 
    if label == 0
        hasil = 'Sehat';
    else
        hasil = 'HFMD';
    end
end
