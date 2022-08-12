%sampling the cest images with full samle scheme
%2019.3.3
%modified 2019.6.22
% created by Angus, wjtcw@hotmail.com
%%%%%%%%%%%%%%%%%%%%%norm1
clc
clear 
close all
load('ogse0728_data-100.mat')
output_dir = 'exp4test/';

row = 128;
col = 128;
inoutC = 16;
outputC = 4;
for slicei = 1:11
    % format
    ogse_image_out = zeros(row,col,inoutC+outputC);
    ogse_image_out(:,:,1:5) = squeeze(ogseN1_mat(slicei,:,:,:));
    ogse_image_out(:,:,6:9) = squeeze(ogseN2_mat(slicei,:,:,:));
    ogse_image_out(:,:,10:16) = squeeze(pgse_mat(slicei,:,:,:));
    ogse_image_out(:,:,17:end) =0;

    ogse_image_out1 = zeros(row,col,inoutC+outputC);
    % resize
    for loopi = 1:inoutC+outputC
        temp = ogse_image_out(:,:,loopi);
        temp = imresize(temp,[row,col],'nearest');
        ogse_image_out1(:,:,loopi) = temp;
    end
    %% mask
    mask = ogse_image_out1(:,:,1);
    mask = mask./max(mask(:));
    th = graythresh(mask)/2;
    mask = imbinarize(mask,th);
    imshow(mask,[])
    % normalize
    % ogsen1
    for loopi = 5:-1:1
        ogse_image_out1(:,:,loopi) = ogse_image_out1(:,:,loopi)./(ogse_image_out1(:,:,1)+eps).*mask;
    end
    % ogsen2
    for loopi = 9:-1:6
        ogse_image_out1(:,:,loopi) = ogse_image_out1(:,:,loopi)./(ogse_image_out1(:,:,6)+eps).*mask;
    end
    % pgse
    for loopi = 16:-1:10
        ogse_image_out1(:,:,loopi) = ogse_image_out1(:,:,loopi)./(ogse_image_out1(:,:,10)+eps).*mask;
    end  
    ogse_image_out = ogse_image_out1;

    %% show data
    for loopi = 1:inoutC
        subplot(5,5,loopi)
        imshow(ogse_image_out(:,:,loopi),[0,1]);colormap jet
    end
    subplot(5,5,loopi+1)
    imshow(ogse_image_out(:,:,loopi+1),[0,30]);colormap jet
    subplot(5,5,loopi+2)
    imshow(ogse_image_out(:,:,loopi+2),[0,0.6]);colormap jet
    subplot(5,5,loopi+3)
    imshow(ogse_image_out(:,:,loopi+3),[0,3]);colormap jet
    %% save
    output =ogse_image_out;
    output = permute(output,[3,1,2]);
    filename1=[output_dir,'brain',num2str(slicei),'.Charles'];
    [fid,msg]=fopen(filename1,'wb');
    fwrite(fid, output,'single');
    fclose(fid); 
end
