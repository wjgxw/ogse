clear variables ; clear obj ; 
addpath(genpath('../mati'))
load('S11_000.mat')
%% Generate DiffusionPulseSequence objects
% Create a trapezoidal cosine OGSE pulse sequence object for all OGSE acquisitions
%% Generate DiffusionPulseSequence objects
% Create a trapezoidal cosine OGSE pulse sequence object for all OGSE acquisitions
ogseN1_bvalue = ogseN1_bvalue/1000;
ogseN2_bvalue = ogseN2_bvalue/1000;
pgse_bvalue = pgse_bvalue/1000;

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
nmodel = 3 ; 
switch nmodel
    case 1, structure.modelName = '1compt' ; structure.geometry = 'sphere' ; 
    case 2, structure.modelName = 'impulsed_vin_d_Dex' ; structure.Din = 1.56 ; structure.betaex = 0 ; structure.geometry = 'sphere';
    case 3, structure.modelName = 'impulsed_vin_d_Dex_Din' ; %structure.betaex = 0 ; structure.geometry = 'sphere';
    case 4, structure.modelName = 'impulsed_vin_d_Dex_Din_betaex' ; %structure.geometry = 'sphere';
end

% Create an IMPULSED model object
impulsed = mati.IMPULSED(structure, pulse) ; 


%% Example of synthesize dMRI signals based on the IMPULSED model
% This is for computer simulations studies to synthesize dMRI signals based on the IMPULSED model.
% The ground-truth microstructural parameter are determined below. 
% NOTE:  
%%
% 
% # parms_sim is a cell array that contains all microstructual parameters for dMRI signal synthesis 
% # variables (*_sim) indicate ground-truth microstructural parameters used in the simulations. 
% 

switch nmodel
    case 1      % [d, Din]
        d = [10:15] ; Din = [1.56 3] ; 
        parms_sim = {d, Din};    [d,Din]=meshgrid(d,Din) ; 
        d_sim = d(:)' ; Din_sim = Din(:)' ; 
    case 2      % [vin, d, Dex]
        vin = [0:0.01:1]  ; d =[4:0.4:40]  ; Dex = [0:0.03:3] ; 
        parms_sim = {vin, d, Dex};   [vin, d,Dex] = meshgrid(vin, d,Dex) ; 
        d_sim = d(:)' ; vin_sim = vin(:)' ; Dex_sim = Dex(:)' ; 
    case 3      % [vin, d, Dex, Din]
        vin = [0.04:0.01:1] ; d = [2:0.5:50] ; Dex = [0.12:0.03:3] ; Din = [1.56] ; 
        parms_sim = {vin, d, Dex, Din};   [vin, d,Dex,Din] = ndgrid(vin, d,Dex,Din) ; 
        d_sim = d(:)' ; vin_sim = vin(:)' ; Dex_sim = Dex(:)' ; Din_sim = Din(:)' ; 
    case 4      % [vin, d, Dex, Din, betaex]
        vin = [0.6] ; d = [8:2:16] ; Dex = [2] ; Din = [1.56] ; betaex = [5] ; 
        parms_sim = {vin, d, Dex, Din, betaex};   [vin, d,Dex,Din,betaex] = ndgrid(vin, d,Dex,Din,betaex) ; 
        d_sim = d(:)' ; vin_sim = vin(:)' ; Dex_sim = Dex(:)' ; Din_sim = Din(:)' ; betaex_sim = betaex(:)' ; 
end

% Synthesize IMPULSED signals based on the microstructural parameters determined above
signal_sim = impulsed.FcnSignal(parms_sim, impulsed); 
% Din_sim = 2*ones(size(Dex_sim));
signal_sim = [signal_sim;d_sim;vin_sim;Dex_sim;Din_sim];
% % Add Rician noise to synthesized signals
% sigma_noise = 0.02 ;         % standard deviation of Gaussian noise in the real and the imaginary images assuming to be equal
% signal_sim = mati.Physics.AddRicianNoise(signal_sim, sigma_noise) ; 
save 'signal_sim_model4' 'signal_sim'
