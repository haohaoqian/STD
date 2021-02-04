function [time_21,tag] = corr_fun(mic_1,mic_2)
% CORR �������
%   �������ȷ��ʱ���
[N1 , ~]= size(mic_1);
[N2 , ~]= size(mic_2);
T1 = N1/20000;%���ļ��ж�ȡ��Fs=20000
T2 = N2/20000;

s=fft(xcorr(mic_1,mic_2,'unbiased'));
r12=ifft(s./abs(s));

t12 = linspace(-T1,T2,N1+N2+1);
[~, p12] = max (r12(5000 : end-5000));%����unbiased��������ƫ��֮�������������̣��ֶ��е���
time_21=-t12(p12+4999);%time2-time1(s)
tag=1;
if time_21>0.2/343
   tag=0;
end
end