function [ogse_image_out,new_mask,ogse_effect_out] =  WJGshape_ogse(ogse_effect,ogse_image,mask,dirname,dirs,row,col,type,ratio,WJGi)
%generate the random shapes, and add texture in the shapes

%mask: the generateed random shape mask
%dir: the directory of the images, which used for generating random textures
%2021.10.28
% created by Angus, wjtcw@hotmail.com
global signal_sim

radius_limit = row;
mini_size = 3;% the minimum size
ogse_image_out = ogse_image;
ogse_effect_out = ogse_effect;
switch type 
    case 1    
        %circle
        mask1 = zeros(row,col);
        RADIUS = randi([mini_size,round(radius_limit/12)],1); 
        center = randi([5,row-5],1,2); 
        temp_mask = WJGgenCircle(row,RADIUS,center);
        if(sum(temp_mask(:))>5)    %prevent the small shape
            mask1 = mask1+temp_mask;
        end
        new_mask = mask1.*abs(mask1-mask);%prevent the overlap between the masks
        [ogse_image_out,ogse_effect_out] = Add_tex(ogse_effect,ogse_image,dirname,dirs,new_mask,ratio,WJGi);  
        new_mask = abs(new_mask+mask)>0;  

    case 2
        %ring
        mask1 = zeros(row,col);
        RADIUS1 = randi([mini_size,round(radius_limit/4)],1);
        RADIUS2 = randi([mini_size,round(radius_limit/16)],1);
        center = randi([-5,5],1,2); 
        temp_mask = WJGgenRing( row,max(RADIUS1,RADIUS2),min(RADIUS1,RADIUS2),center );
        if(sum(temp_mask(:))>5)    
            mask1 = mask1+temp_mask;
        end
        new_mask = mask1.*abs(mask1-mask); 
        [ogse_image_out,ogse_effect_out] = Add_tex(ogse_effect,ogse_image,dirname,dirs,new_mask,ratio,WJGi);  

        new_mask = abs(new_mask+mask)>0; 
    case 3
        %square
        mask1 = zeros(row,col);
        shape = randi([1,round(radius_limit/8)],2,2);  
        center = randi([5,row-row/8],2,1); 
        center = [center,center];
        shape = shape+center;
        x = min([sort(shape(1,:));row,row]);   
        y = min([sort(shape(2,:));col,col]);
        vx = [x(1),x(1),x(2),x(2),x(1)];
        vy = [y(1),y(2),y(2),y(1),y(1)];
        temp_mask = WJG_convex_S(row,vx,vy);
        if(sum(temp_mask(:))>5)   
            mask1 = mask1+temp_mask;
        end
        new_mask = mask1.*abs(mask1-mask); 
        [ogse_image_out,ogse_effect_out] = Add_tex(ogse_effect,ogse_image,dirname,dirs,new_mask,ratio,WJGi); 
        new_mask = abs(new_mask+mask)>0;
    case 4 
        %triangle
        mask1 = zeros(row,col);
        shape = randi([1,round(radius_limit/8)],2,3);
        center = randi([5,row-row/8],2,1); 
        center = [center,center,center];
        shape = shape+center;
        x = min([sort(shape(1,:));row,row,row]);
        y = min([sort(shape(2,:));col,col,col]);
        vx = [x(1),x(2),x(3),x(1)];
        vy = [y(1),y(2),y(3),y(1)];
        temp_mask = WJG_convex_S(row,vx,vy);
        if(sum(temp_mask(:))>5)     
            mask1 = mask1+temp_mask;
        end
        new_mask = mask1.*abs(mask1-mask); 
        [ogse_image_out,ogse_effect_out] = Add_tex(ogse_effect,ogse_image,dirname,dirs,new_mask,ratio,WJGi);  
        new_mask = abs(new_mask+mask)>0;
end
      
% imshow(mask1+mask2+mask3+mask4)

function [ogse_image_out,ogse_effect_out] =  Add_tex(ogse_effect,ogse_image,dirname,dirs,new_mask,ratio,WJGi)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global signal_sim slice output_channel
ogse_image_out = ogse_image;
% [row,col,slice] = size(ogse_image);

% samples = length(dirs);
% frame_i = randi([1,samples],1); 
% filename = [dirname,dirs(frame_i).name];
% II1 = double(rgb2gray(imread(filename)))/255;% 
% II1 = abs(imresize(II1,[row,col],'nearest'));
% w = fspecial('gaussian',4,4);
% II1 = imfilter(II1,w','replicate');
% %%2
% frame_i = randi([1,samples],1); 
% filename = [dirname,dirs(frame_i).name];
% II2 = double(rgb2gray(imread(filename)))/255;% 
% II2 = abs(imresize(II2,[row,col],'nearest'));
% w = fspecial('gaussian',4,4);
% II2 = imfilter(II2,w','replicate');
% %%3
% frame_i = randi([1,samples],1); 
% filename = [dirname,dirs(frame_i).name];
% II3 = double(rgb2gray(imread(filename)))/255;% 
% II3 = abs(imresize(II3,[row,col],'nearest'));
% w = fspecial('gaussian',4,4);
% II3 = imfilter(II3,w','replicate');
% %%4
% frame_i = randi([1,samples],1); 
% filename = [dirname,dirs(frame_i).name];
% II4 = double(rgb2gray(imread(filename)))/255;% 
% II4 = abs(imresize(II4,[row,col],'nearest'));
% w = fspecial('gaussian',4,4);
% II4 = imfilter(II4,w','replicate');

decay = rand()*0.8+0.2;

S_temp = signal_sim(:,WJGi);
% plot(S_temp(1:slice))
% 
%% input
for loopi = 1:slice 
    %select_b = floor(length(bvalue)/(slice-1))*(loopi-1)+1;
    temp_ogse= (new_mask.*1*ratio+(new_mask*(1-ratio))).*S_temp(loopi)*decay;
    ogse_image_out(:,:,loopi) = temp_ogse+ogse_image(:,:,loopi);
end

%% out 
d = S_temp(slice+1);
vin = S_temp(slice+2);
Dex = S_temp(slice+3);
Din = S_temp(slice+4);

d = (new_mask.*1*ratio+(new_mask*(1-ratio))).*d;
vin = (new_mask.*1*ratio+(new_mask*(1-ratio))).*vin;
Dex = (new_mask.*1*ratio+(new_mask*(1-ratio))).*Dex;
Din = (new_mask.*1*ratio+(new_mask*(1-ratio))).*Din;
ogse_effect_out(:,:,1) = d+ogse_effect(:,:,1);
ogse_effect_out(:,:,2) = vin+ogse_effect(:,:,2);
ogse_effect_out(:,:,3) = Dex+ogse_effect(:,:,3);
ogse_effect_out(:,:,4) = Din+ogse_effect(:,:,4);

