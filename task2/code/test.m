clear;close all;clc;

test_dir='..\test\';
fid=fopen([test_dir,'\result.txt'],'w'); 

time_table=[];
for angle=0:359
    [time_21,time_31,time_41,time_32,time_42,time_43] = angle2time(angle);
    time_table=[time_table;time_21,time_31,time_41,time_32,time_42,time_43];
end

%测试
for num=1:140
    mic_1=audio_process(audioread([test_dir,num2str(num),'_mic1.wav']));
    mic_2=audio_process(audioread([test_dir,num2str(num),'_mic2.wav']));
    mic_3=audio_process(audioread([test_dir,num2str(num),'_mic3.wav']));
    mic_4=audio_process(audioread([test_dir,num2str(num),'_mic4.wav']));
    T_list=corr(mic_1,mic_2,mic_3,mic_4);
    angle=orient(T_list,time_table);
    fprintf(fid,'%f\n',angle);
end
fclose(fid);