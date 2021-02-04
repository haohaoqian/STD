function [time_21,tag] = corr_fun(mic_1,mic_2)
% CORR 广义相关
%   广义相关确定时间差
[N1 , ~]= size(mic_1);
[N2 , ~]= size(mic_2);
T1 = N1/20000;%从文件中读取的Fs=20000
T2 = N2/20000;

s=fft(xcorr(mic_1,mic_2,'unbiased'));
r12=ifft(s./abs(s));

t12 = linspace(-T1,T2,N1+N2+1);
[~, p12] = max (r12(5000 : end-5000));%经过unbiased消除中心偏差之后两端容易上翘，手动切掉！
time_21=-t12(p12+4999);%time2-time1(s)
tag=1;
if time_21>0.2/343
   tag=0;
end
end