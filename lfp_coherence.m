%LFP_COHERENCE   Compute coherence between two signals. 
%   LFP_COHERENCe(X,Y) computes the coherence between the two time varrying
%   signals. This simple script allows me to call this function from
%   python. Thus, giving me access to the powerful Chronux toolbox!
%
% G. Telian
% Adesnik Lab
% 20180216


function [Cxy,f, Cerr] = lfp_coherence(x, y)
% % define chronux parameters
params.tapers = [2,5]; params.Fs = 1500; params.err = [2, 0.05]; params.trialave = 1;

% compute coherence
[Cxy, phi, S12, S1, S2, f, confCxy, phistd, Cerr] = coherencyc(x, y, params);
