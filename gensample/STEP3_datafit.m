%generate ivim images
%2021.10.28
% created by Angus, wjtcw@hotmail.com
clc
clear 
close all
warning off
addpath(genpath('D:\Users\angus\Desktop\other\ogse\mati'))
load('gensample_model3/1.mat')
load('zhangyuzhenS11_000.mat')
ogseN1_mat(1,:,:,:) = ogse_image_out(:,:,1:5);
ogseN2_mat(1,:,:,:) = ogse_image_out(:,:,6:9);
pgse_mat(1,:,:,:) = ogse_image_out(:,:,10:16);

maskname = 'mask.mat';
slicei=1;
exclude_b0=1;
loadmask = 0;
wholebrain=1;
out_set = [6;10]; %if the later signal intensity id biger then the before, then set the value to the before value
ext_dim=128;
%% Generate DiffusionPulseSequence objects
% Create a trapezoidal cosine OGSE pulse sequence object for all OGSE acquisitions
if (exclude_b0==1)
    ogseN1_bvalue=ogseN1_bvalue(2:end)/1000;
    ogseN2_bvalue=ogseN2_bvalue(2:end)/1000;
    pgse_bvalue=pgse_bvalue(2:end)/1000;
else
    ogseN1_bvalue=ogseN1_bvalue/1000;
    ogseN2_bvalue=ogseN2_bvalue/1000;
    pgse_bvalue=pgse_bvalue/1000;
end
Nacq1 = length(ogseN1_bvalue) ;          % total number of acquisition points
pulse_tcos1 = mati.DiffusionPulseSequence(Nacq1,...
    'TE',               TE,...                 % echo time [ms]
    'delta',            ogse_N1_delta,...                  % gradient duration [ms]
    'Delta',            ogse_N1_DELTA,...                  % separation of two gradients [ms]
    'b',                  ogseN1_bvalue, ...      % b value [ms/um^2]
    'n',                  ones(1,Nacq1),...             % number of oscillating cycles
    'shape',         "tcos",...                 % gradient waveform shape
    'gdir',             select_ori',...              % gradient directions. It should be a Nx3 matrix
    'trise',            ogse_N1_trise) ;                  % gradient rise time [ms]
Nacq2 = length(ogseN2_bvalue) ;          % total number of acquisition points
pulse_tcos2 = mati.DiffusionPulseSequence(Nacq2,...
    'TE',               TE,...                 % echo time [ms]
    'delta',            ogse_N2_delta,...                  % gradient duration [ms]
    'Delta',            ogse_N2_DELTA,...                  % separation of two gradients [ms]
    'b',                  ogseN2_bvalue, ...      % b value [ms/um^2]
    'n',                  2*ones(1,Nacq2),...             % number of oscillating cycles
    'shape',         "tcos",...                 % gradient waveform shape
    'gdir',             select_ori',...              % gradient directions. It should be a Nx3 matrix
    'trise',            ogse_N2_trise) ;                  % gradient rise time [ms]
% Create a trapezoidal PGSE pulse sequence object for all PGSE acquisitions

Nacq3 = length(pgse_bvalue) ; 
pulse_tpgse = mati.DiffusionPulseSequence(Nacq3, ...
    'TE',               TE,...
    'delta',            pgse_delta, ...
    'Delta',            pgse_DELTA, ...
    'b',                  pgse_bvalue, ...
    'shape',         "tpgse",...
    'gdir',             select_ori',...
    'trise',             pgse_trise) ; 

% Combine OGSE and PGSE pulse sequence objects
pulse = mati.PulseSequence.cat(pulse_tcos1,pulse_tcos2, pulse_tpgse) ; 
% An example of choosing a subset of PulseSequence object to meet e.g., hardware limitations
pulse = pulse(pulse.G<80e-5) ;      % 80mT/m = 80 x 10^(-5) gauss/um
% Display the PulseSequence object
pulse.disp(pulse)


%% Generate IMPULSED model object
% Choose which specific model to use. Note that IMPULSED can fit up to five parameters, i..e, $d$, $v_{in}$, $D_{in}$, $D_{ex0}$, and $\beta_{ex}$. 
% Individual parameters could be fixed during fitting to enhance the fitting precision of other parameters. 

nmodel = 2 ; 
switch nmodel
    case 1, structure.modelName = '1compt' ; structure.geometry = 'sphere' ; 
    case 2, structure.modelName = 'impulsed_vin_d_Dex' ; structure.Din = 2 ; structure.betaex = 0 ; structure.geometry = 'sphere';
    case 3, structure.modelName = 'impulsed_vin_d_Dex_Din' ; %structure.betaex = 0 ; structure.geometry = 'sphere';
    case 4, structure.modelName = 'impulsed_vin_d_Dex_Din_betaex' ; %structure.geometry = 'sphere';
end
% structure.geometry='cylinder';
% Create an IMPULSED model object
impulsed = mati.IMPULSED(structure, pulse) ; 


%% regieter 
[nslice,~,~,nbvalue] = size(pgse_mat);
pgse_mat1 = zeros(nslice,ext_dim,ext_dim,nbvalue);
for loopi = 1:nbvalue
    tempI  =  squeeze(pgse_mat(slicei,:,:,loopi));
    tempI = imresize(tempI,[ext_dim,ext_dim],'nearest');
    pgse_mat1(slicei,:,:,loopi) = tempI;
end
pgse_mat = pgse_mat1;

[nslice,~,~,nbvalue] = size(ogseN1_mat);
ogseN1_mat1 = zeros(nslice,ext_dim,ext_dim,nbvalue);
for loopi = 1:nbvalue
    tempI  =  squeeze(ogseN1_mat(slicei,:,:,loopi));
    tempI = imresize(tempI,[ext_dim,ext_dim],'nearest');
    ogseN1_mat1(slicei,:,:,loopi) = tempI;
end    
ogseN1_mat = ogseN1_mat1;

[nslice,~,~,nbvalue] = size(ogseN2_mat);
ogseN2_mat1 = zeros(nslice,ext_dim,ext_dim,nbvalue);
for loopi = 1:nbvalue
    tempI  =  squeeze(ogseN2_mat(slicei,:,:,loopi));
    tempI = imresize(tempI,[ext_dim,ext_dim],'nearest');
    ogseN2_mat1(slicei,:,:,loopi) = tempI;
end   
ogseN2_mat = ogseN2_mat1;

%% mask
if wholebrain ==0
    if loadmask == 1
        load(maskname)
    else
        mask = squeeze(ogseN1_mat(slicei,:,:,1));
        imshow(mask,[]);colormap jet
        mask = roipoly();
        save(maskname,'mask')
    end
else
	mask = squeeze(ogseN1_mat(slicei,:,:,1));
    mask = mask./max(mask(:));
    th = graythresh(mask)/10;
    mask = imbinarize(mask,th);
    mask = bwareaopen(mask,1000,4);
    imshow(mask,[])
end
imshow(mask,[]);
idx = find(mask==1);

%% normalized
ogseN1_mat = squeeze(ogseN1_mat(slicei,:,:,:));
ogseN2_mat = squeeze(ogseN2_mat(slicei,:,:,:));
pgse_mat = squeeze(pgse_mat(slicei,:,:,:));
% ogseN1_mat = ogseN1_mat./squeeze(ogseN1_mat(:,:,1)+eps);
% ogseN2_mat = ogseN2_mat./squeeze(ogseN2_mat(:,:,1)+eps);
% pgse_mat = pgse_mat./squeeze(pgse_mat(:,:,1)+eps);
signal_sim = zeros(Nacq1+Nacq2+Nacq3+exclude_b0*3,length(idx));
for loopi = 1:length(idx)
    [idx_row,idx_col] = ind2sub([ext_dim,ext_dim],idx(loopi));
    temp = [squeeze(ogseN1_mat(idx_row,idx_col,:));squeeze(ogseN2_mat(idx_row,idx_col,:));squeeze(pgse_mat(idx_row,idx_col,:))];
    signal_sim(:,loopi) = temp;
end
%% remove the strange data
idx_out = [];
for loopi = 1:size(signal_sim,2)
    temp = signal_sim(:,loopi);  
    temp_diff = diff(temp);
    idx1 = find(temp_diff>0)+1;
    idx2 = setdiff(idx1, out_set);
    for loopj = 1:length(idx2)
        signal_sim(idx2(loopj),loopi) = signal_sim(idx2(loopj)-1,loopi);
    end
    if (abs(temp(1)-temp(6))/temp(1)>0.2)
        idx_out = [idx_out,loopi];
    end
end
idx(idx_out)=[];
signal_sim(:,idx_out)=[];
%% normalized
temp1 = signal_sim(1:Nacq1+exclude_b0,:);
temp1 = temp1./max(temp1);

temp2 = signal_sim(Nacq1+1+exclude_b0:Nacq1+Nacq2+exclude_b0*2,:);
temp2 = temp2./max(temp2);

temp3 = signal_sim(Nacq1+Nacq2+1+exclude_b0*2:end,:);
temp3 = temp3./max(temp3);
% plot(signal_sim)
%% denoise SVD
signal_sim = [temp1;temp2;temp3];
select = floor(16);
[UA,SAmat,VA] = svd(signal_sim,'econ');
SAmat(select+1:end,select+1:end) = 0;
signal_sim = UA*SAmat*transpose(VA);
temp1 = signal_sim(1:Nacq1+exclude_b0,:);
temp1 = temp1./max(temp1);
temp2 = signal_sim(Nacq1+1+exclude_b0:Nacq1+Nacq2+exclude_b0*2,:);
temp2 = temp2./max(temp2);
temp3 = signal_sim(Nacq1+Nacq2+1+exclude_b0*2:end,:);
temp3 = temp3./max(temp3);
signal_sim = [temp1;temp2;temp3];
% plot(signal_sim)
if (exclude_b0==1)
    signal_sim([1,6,10],:) = [];
end
% Create an ImageData object
sigma_noise = 0;
[Npulse, Nparms] = size(signal_sim) ; 
data = mati.ImageData(reshape(signal_sim',[Nparms, 1, 1, Npulse]), sigma_noise) ; 

%% Fit IMPULSED model to dMRI signals
% Create a Fit object
fitopts.solverName = 'lsqnonlin'; % {'lsqcurvefit' , 'lsqnonlin' , 'fmincon'}
fitopts.options = optimoptions(fitopts.solverName,'Display','off') ; 
fitopts.noiseModel = 'standard' ; %{'none','standard';'logLikelihood'}
fitopts.flag.parfor = 'y' ;              % If use parallel computing with parfor
fitopts.flag.deivim = 'n' ;             % if remove IVIM influence
fitopts.flag.multistart = 'y' ;         % If try fittings multiple times with different initial conditions
fitopts.NumStarts = 1 ;              % if try multistart=='y', try how many times of different initial conditions? 

% Create a data fitting object
fitpars = mati.FitPars(impulsed, fitopts) ; 
warning off ; 

% Fit model to data
tic
fitout = fitpars.Fit(data) ; 
toc
%% show the maps
dmean_map = zeros(ext_dim,ext_dim);
vin_map = zeros(ext_dim,ext_dim);
dex_map = zeros(ext_dim,ext_dim);
sta_data = fitout.d(:);
vin = fitout.vin(:);
Dex = fitout.Dex(:);
for loopi = 1:length(idx)
    [idx_row,idx_col] = ind2sub([ext_dim,ext_dim],idx(loopi));
    dmean_map(idx_row,idx_col)=sta_data(loopi);
    vin_map(idx_row,idx_col)=vin(loopi);
    dex_map(idx_row,idx_col)=Dex(loopi);

end
figure(1);imshow(dmean_map,[0,25]);colormap jet;
figure(2);imshow(vin_map,[0,0.35]);colormap jet;
figure(3);imshow(dex_map,[0,2.5]);colormap jet;
    
figure(4);imshow(ogse_image_out(:,:,17),[0,25]);colormap jet;
figure(5);imshow(ogse_image_out(:,:,18),[0,0.35]);colormap jet;
figure(6);imshow(ogse_image_out(:,:,19),[0,2.5]);colormap jet;  

