%generate ivim images
%2021.10.28
% created by Angus, wjtcw@hotmail.com
clc
clear 
close all
warning off
addpath(genpath('../mati'))
addpath(genpath('../MRiLab_master'))
global signal_sim output_channel slice
load('signal_sim_model4.mat')
[~,num_signal] = size(signal_sim);
idx = randperm(num_signal);
signal_sim = signal_sim(:,idx);
row = 256;
col = 256;
output_channel = 4; 
slice = size(signal_sim,1)-output_channel;
num = 400;      %the number of mask
ratio = 0; %the ratio of texture
samplenum = 1000; %the number of samples we want to generate
dirname = 'image/';
dirs=dir([dirname,'*.jpg']);
temp_brain_mask = zeros(row,col);% 
temp_brain_mask(10:row-10,10:col-10) = 1;
temp_brain_mask = repmat(temp_brain_mask,1,1,slice+output_channel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('shuffle');%the seed of random
WJGi = 1;
for loopj = 1:samplenum   
    ogse_image_out=zeros(row,col,slice);
    ogse_mask = zeros(row,col);
    ogse_effect = zeros(row,col,output_channel);
    for loopk = 1:num
        [ogse_image_out,ogse_mask,ogse_effect] =  WJGshape_ogse(ogse_effect,ogse_image_out,ogse_mask,dirname,dirs,row,col,1,ratio,WJGi);
        WJGi = mod(WJGi,num_signal-1)+1;
        [ogse_image_out,ogse_mask,ogse_effect] =  WJGshape_ogse(ogse_effect,ogse_image_out,ogse_mask,dirname,dirs,row,col,2,ratio,WJGi);
        WJGi = mod(WJGi,num_signal-1)+1;
        [ogse_image_out,ogse_mask,ogse_effect] =  WJGshape_ogse(ogse_effect,ogse_image_out,ogse_mask,dirname,dirs,row,col,3,ratio,WJGi);
        WJGi = mod(WJGi,num_signal-1)+1;
        [ogse_image_out,ogse_mask,ogse_effect] =  WJGshape_ogse(ogse_effect,ogse_image_out,ogse_mask,dirname,dirs,row,col,4,ratio,WJGi);
        WJGi = mod(WJGi,num_signal-1)+1;
        if (sum(ogse_mask(:))>row*col*0.9) 
            break;
        end

    end
    ogse_image_out(:,:,slice+1:slice+output_channel) = ogse_effect;
    ogse_image_out = ogse_image_out.*temp_brain_mask; 

%     for loopi = 1:slice
%         subplot(5,5,loopi)
%         imshow(ogse_image_out(:,:,loopi),[0,1]);colormap jet
%     end
%         subplot(5,5,loopi+1)
%         imshow(ogse_image_out(:,:,loopi+1),[0,30]);colormap jet
%         subplot(5,5,loopi+2)
%         imshow(ogse_image_out(:,:,loopi+2),[0,0.6]);colormap jet
%         subplot(5,5,loopi+3)
%         imshow(ogse_image_out(:,:,loopi+3),[0,3]);colormap jet
    
    filenames =['gensample_model3/',num2str(loopj),'.mat'];
    save(filenames,'ogse_image_out','-single');
    loopj
end


