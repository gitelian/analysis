%LFP_PSD Compute PSD.
%   LFP_PSD(X) computes the PSD of the time varying LFP signal.
%   This simple script allows me to call this function from
%   python. Thus, giving me access to the powerful Chronux toolbox!
%
% G. Telian
% Adesnik Lab
% 20180216


function [S, f, Serr] = lfp_psd(x)

% % define chronux parameters
params.tapers = [2,5]; params.Fs = 1500; params.err = [2, 0.05]; params.trialave = 1;

[S, f, Serr] = mtspectrumc(x, params);