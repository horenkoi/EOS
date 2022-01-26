clear all
close all

rand('seed',1)
randn('seed',1)
LASTN = maxNumCompThreads(1)
FS=18;

N_ann=20;
T=2000;
DD=100;% 250 300 400 600 800 1000];
EPS=[0.09 0.1:0.05:1 2 4];
%EPS=[0.8 0.8 0.7 0.6 0.5 0.5 0.5 0.4 0.3 0.3 0.2 0.2 0.3 0.4 0.4];%[400 380 280 250 180 120 110 100 150 180 200]/320;% 80 80 90 100 110 130];
%T=2500;
%DD=[20 50 100 150 200 250 300 350 400 450 500 600 700 800 1000 1200];
%EPS=[150 120 110 100 90 80 80 80 80 80 80 80 80 80 90 100];
prop=0.02;
n_ann=1;
for ind_T=1:length(EPS)
    d=DD;
    T_valid=floor(3/4*T);

    A=[[randn(1,T-prop*T);randn(d-1,T-prop*T)] 5*(rand(d,round(prop*T)))+1];%0.1*randn(d,T+round(prop*T))];
    A_valid=[[randn(1,round(T_valid-prop*T_valid));randn(d-1,round(T_valid-prop*T_valid))] 5*(rand(d,round(prop*T_valid)))+1];%0.1*randn(d,T_valid+round(prop*T_valid))];
    outl_patern=[0*randn(1,T-prop*T) ones(1,round(prop*T))];
    tic;
    [w_out(ind_T,:),N_iter(n_ann,ind_T),mu_out,V_out,L_train,ind_out]=EDO(A,EPS(ind_T),1,A_valid);
    t(n_ann,ind_T)=toc;
    stats = confusionmatStats(outl_patern,double(ind_out));
    ACC_EOD(n_ann,ind_T)=stats.precision(2)
    F1_EOD(n_ann,ind_T)=stats.Fscore(2)
    figure(151);clf;hold on;box on;
    ii=find(outl_patern==0);plot(A(1,ii),A(2,ii),'b.','MarkerSize',2);
    ii=find(outl_patern==1);plot(A(1,ii),A(2,ii),'bx','LineWidth',3,'MarkerSize',9);
    ii=find(ind_out==1);plot(A(1,ii),A(2,ii),'ro','LineWidth',2,'MarkerSize',9);
    xlabel('dimension 1','FontSize',FS); ylabel('dimension 2','FontSize',FS);
    legend({'normally-distributed samples (\mu_d=0)','uniformly distributed outliers (\mu_d=3.5)','outliers detected with EOS'},'FontSize',18);
    title({['Outlier detection with Entropic Outlier Sparsification algorithm (EOS)']
        ['(T=' num2str(size(A,2)) ', D=' num2str(d) ', p=' num2str(prop) ')']});set(gcf,'Position',[10 100 1300  700]);
    set(gca,'FontSize',24,'LineWidth',2);
end

[X,Y]=meshgrid(EPS,1:T);

figure;surfl(X,Y,w_out');shading interp;
title({['Dependencer of EOS solutions w^\alpha on regularization parameter \alpha'],['(T=' num2str(size(A,2)) ', D=' num2str(d) ')']});set(gcf,'Position',[10 100 1300  700]);
xlabel('Regularization Parameter \alpha','FontSize',18); ylabel('Data Statistics index t','FontSize',FS);zlabel('EOS weights w_t^\alpha','FontSize',18);
xlim([0.08 10])
set(gca,'FontSize',24,'LineWidth',2,'XScale','log','ZScale','log');
