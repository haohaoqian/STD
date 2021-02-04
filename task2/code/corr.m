function T_list=corr(mic_1,mic_2,mic_3,mic_4)
%CORR 此处显示有关此函数的摘要
%   此处显示详细说明
[time_21,tag_21]=corr_fun(mic_1,mic_2);
[time_31,tag_31]=corr_fun(mic_1,mic_3);
[time_41,tag_41]=corr_fun(mic_1,mic_4);
[time_32,tag_32]=corr_fun(mic_2,mic_3);
[time_42,tag_42]=corr_fun(mic_2,mic_4);
[time_43,tag_43]=corr_fun(mic_3,mic_4);

T_list=[time_21,tag_21;time_31,tag_31;time_41,tag_41;time_32,tag_32;time_42,tag_42;time_43,tag_43];

end