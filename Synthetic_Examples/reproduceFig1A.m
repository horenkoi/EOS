clear all
close all

load illia_test_prop.mat

%% Reproducing Figure 1A

MinMarkSize=5;
MaxMarkSize=250;
OutMarkSize=0.2;
lag=2;
ii1=find(w_out<1e-5);ii2=find(w_out>1e-5);
ms(ii1)=MinMarkSize;ms(ii2)=w_out(ii2)-min(w_out(ii2));ms(ii2)=w_out(ii2)./max(w_out(ii2));
ms(ii2)=(MaxMarkSize-MinMarkSize)*ms(ii2)+MinMarkSize;
figure;scatter(A(1,1:lag:T),A(2,1:lag:T),ms(1:lag:T),'filled','LineWidth',2); 
hold on;plot(A(1,ii1(1:lag:length(ii1))),A(2,ii1(1:lag:length(ii1))),'ro','MarkerSize',5); 
