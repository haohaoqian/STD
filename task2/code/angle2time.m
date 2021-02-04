function [time_21,time_31,time_41,time_32,time_42,time_43] = angle2time(angle)
%ANGLE2TIME 此处显示有关此函数的摘要
%   此处显示详细说明
v_sonic=343;
d=0.2;
d1=0.1*sqrt(2);
if angle<90
    time_21=d1*sin(pi*(45-angle)/180)/v_sonic;
    time_41=-d1*sin(pi*(-angle-45)/180)/v_sonic;
elseif angle<180
    time_21=d1*sin(pi*(45-angle)/180)/v_sonic;
    time_41=-d1*sin(pi*(angle-135)/180)/v_sonic;
elseif angle<270
    time_21=d1*sin(pi*(angle-225)/180)/v_sonic;
    time_41=-d1*sin(pi*(angle-135)/180)/v_sonic; 
else
    time_21=d1*sin(pi*(angle-225)/180)/v_sonic;  
    time_41=-d1*sin(pi*(315-angle)/180)/v_sonic;
end
time_31=d*cos(pi*angle/180)/v_sonic;
time_32=time_31-time_21;
time_42=time_41-time_21;
time_43=time_41-time_31;
end

