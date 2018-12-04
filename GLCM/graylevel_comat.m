function [hasilanalisis] = graylevel_comat(input)

%     gray level (skala 1-8)
    range = getrangefromclass(input);
    slope = 8 / (range(2)-range(1));
    intercept = 1 - (slope*(range(1)));
    input = floor(imlincomb(slope,input,intercept,'double'));
    input(input > 8) = 8;
    input(input < 1) = 1;
    
%     glcm, analisis secara horizontal
    output = zeros(8,8);
    for i=1:size(input,1)
        for j=1:size(input,2)-1
            output(input(i,j), input(i,j+1)) = output(input(i,j), input(i,j+1))+1;
        end
    end
    
%     normalisasi
    glcm = output/sum(sum(output));

%     tekstur
    energi = sum(glcm(:).^2);
    entrophy = sum(glcm(:).*mylog2(glcm(:)));
    [kolom, baris] = meshgrid(1:size(glcm,1), 1:size(glcm,2));
    homogen = sum(sum(glcm./(1+abs(baris-kolom))));
    kontras = sum(sum((abs(baris-kolom).^2).*glcm));
    meanbaris = sum(sum(baris.*glcm));
    meankolom = sum(sum(kolom.*glcm));
    stdbaris = sqrt(sum(sum((baris - meanbaris).^2 .* glcm)));
    stdkolom = sqrt(sum(sum((kolom - meankolom).^2 .* glcm)));
    korelasi = sum(sum((baris - meanbaris) .* (kolom - meankolom) .* glcm)) / (stdbaris * stdkolom);
    hasilanalisis = [kontras korelasi energi homogen entrophy];
end

function out = mylog2(in)
  out = log2(in);
  out(~in) = 0;
end