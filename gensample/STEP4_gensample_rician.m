%sampling the cest images with full samle scheme
%2019.3.3
%modified 2019.6.22
% created by Angus, wjtcw@hotmail.com
%%%%%%%%%%%%%%%%%%%%%
clc
clear 
close all
addpath(genpath('mati'))
addpath('func')
addpath('CEST_tool')

row = 128;
col = 128;
slice = 16;
outputC = 4;
sigma = 0.5e-2;
sample_dir = 'gensample_model4/';
output_dir = '/data4/angus_wj/ogse/ogse_deep4/different_snr/data/data4train0.5/';
fid_dir_all = dir([sample_dir,'*.mat']);       %list all files
ogse_image_out1 = zeros(row,col,slice+outputC);
for loopj = 1:length(fid_dir_all)
    fid_file = [sample_dir,fid_dir_all(loopj).name];
    load(fid_file);
    % resize
    for loopi = 1:slice+outputC
        temp = ogse_image_out(:,:,loopi);
        temp = imresize(temp,[row,col],'nearest');
        ogse_image_out1(:,:,loopi) = temp;
    end
    % normalize
    % ogsen1
     
    for loopi = 5:-1:1
        ogse_image_out1(:,:,loopi) = ogse_image_out1(:,:,loopi)./(ogse_image_out1(:,:,1)+eps);
        temp = ogse_image_out1(:,:,loopi);
        ogse_image_out1(:,:,loopi)  = mati.Physics.AddRicianNoise(temp, sigma);
        
    end
    % ogsen2
    for loopi = 9:-1:6
        ogse_image_out1(:,:,loopi) = ogse_image_out1(:,:,loopi)./(ogse_image_out1(:,:,6)+eps);
        temp = ogse_image_out1(:,:,loopi);
        ogse_image_out1(:,:,loopi)  = mati.Physics.AddRicianNoise(temp, sigma);
    end
    % pgse
    for loopi = 16:-1:10
        ogse_image_out1(:,:,loopi) = ogse_image_out1(:,:,loopi)./(ogse_image_out1(:,:,10)+eps);
        temp = ogse_image_out1(:,:,loopi);
        ogse_image_out1(:,:,loopi)  = mati.Physics.AddRicianNoise(temp, sigma);
    end  
    ogse_image_out = ogse_image_out1;
    ogse_image_out(:,:,17) = ogse_image_out(:,:,17)/10;
%% show data
%     for loopi = 1:slice
%         subplot(5,5,loopi)
%         imshow(ogse_image_out(:,:,loopi),[0,1]);colormap jet
%     end
%     subplot(5,5,loopi+1)
%     imshow(ogse_image_out(:,:,loopi+1),[0,3]);colormap jet
%     subplot(5,5,loopi+2)
%     imshow(ogse_image_out(:,:,loopi+2),[0,0.6]);colormap jet
%     subplot(5,5,loopi+3)
%     imshow(ogse_image_out(:,:,loopi+3),[0,3]);colormap jet
%% save
% snr
%     width=4;
%     high =4;
%     signal_pos = [40,38,width,high];
%     noise_pos = [33,124,width,width];
%     temp_I = squeeze(ogse_image_out1(:,:,1));
%     imshow((temp_I),[]);hold on
%     temp_SNR=WJG_cal_snr(signal_pos,noise_pos,temp_I)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    output = ogse_image_out;
    output = permute(output,[3,1,2]);
    filename1=[output_dir, num2str(loopj),'.Charles'];
    [fid,msg]=fopen(filename1,'wb');
    fwrite(fid, output,'single');
    fclose(fid); 
    loopj

end








