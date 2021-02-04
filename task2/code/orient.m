function angle = orient(T_list,time_table)
%ORIENT ��λ
%   ��ʱ���ȷ���Ƕ�
min_error=10;
min_angle=0;
T=[];
index=[];
for row=1:6
    if T_list(row,2)==1
        T=[T,T_list(row,1)];
        index=[index,row];
    end
end
time_table=time_table(:,index);

for angle=1:360
    error=1-sum(((time_table(angle,:)/norm(time_table(angle,:)))).*(T/norm(T)));
    if error<min_error
       min_error=error;
       min_angle=angle-1;
    end
end
angle=min_angle;
end