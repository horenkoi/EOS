clear all
close all

rand('seed',1)
randn('seed',1)
LASTN = maxNumCompThreads(1)
FS=18;

N_ann=10;
T=1000;
DD=[150];% 250 300 400 600 800 1000];
%EPS=[0.8 0.8 0.7 0.6 0.5 0.5 0.5 0.4 0.3 0.3 0.2 0.2 0.3 0.4 0.4];%[400 380 280 250 180 120 110 100 150 180 200]/320;% 80 80 90 100 110 130];
%T=2500;
%DD=[20 50 100 150 200 250 300 350 400 450 500 600 700 800 1000 1200];
%EPS=[150 120 110 100 90 80 80 80 80 80 80 80 80 80 90 100];
PROP=[0.01 0.02 0.05 0.08 0.1 0.15 0.2 0.25 0.3];
EPS=[0.4 0.4 0.4 0.3 0.2 0.2 0.2 0.25 0.25];
for n_ann=1:N_ann
    for ind_T=1:length(PROP)
        d=DD;
        T_valid=floor(3/4*T);
        prop=PROP(ind_T);
        A=[[randn(1,T-prop*T);randn(d-1,T-prop*T)] 5*(rand(d,round(prop*T)))+1];%0.1*randn(d,T+round(prop*T))];
        A_valid=[[randn(1,round(T_valid-prop*T_valid));randn(d-1,round(T_valid-prop*T_valid))] 5*(rand(d,round(prop*T_valid)))+1];%0.1*randn(d,T_valid+round(prop*T_valid))];
        outl_patern=[0*randn(1,T-prop*T) ones(1,round(prop*T))];
        tic;
        [mu,S,RD,RD_valid,chi_crt]=DetectMultVarOutliers(A',A_valid');
        t_hadi(n_ann,ind_T)=toc;
        stats = confusionmatStats(outl_patern,double(RD>chi_crt(4)));
        figure(150);clf;hold on;box on;
        ii=find(outl_patern==0);plot(A(1,ii),A(2,ii),'b.','MarkerSize',2);
        ii=find(outl_patern==1);plot(A(1,ii),A(2,ii),'bx','LineWidth',3,'MarkerSize',9);
        ii=find((RD>chi_crt(4))==1);plot(A(1,ii),A(2,ii),'ro','LineWidth',2,'MarkerSize',6);
        xlabel('dimension 1','FontSize',FS); ylabel('dimension 2','FontSize',FS); 
        legend({'normally-distributed samples (\mu_d=0)','uniformly distributed outliers (\mu_d=3.5)','outliers detected with MH'},'FontSize',16);
        title({['Outlier detection with Multivariate Hadi algorithm (MH)']
        ['(T=' num2str(size(A,2)) ', D=' num2str(d) ', p=' num2str(prop) ')']});set(gcf,'Position',[10 100 1300  700]);
        set(gca,'FontSize',24,'LineWidth',2);
        ACC_hadi(n_ann,ind_T)=stats.precision(2)
        F1_hadi(n_ann,ind_T)=stats.Fscore(2)
        tic;
        [w_out,N_iter(n_ann,ind_T),mu_out,V_out,L_train,ind_out]=EDO(A,EPS(ind_T),1,A_valid);
        t(n_ann,ind_T)=toc;
        stats = confusionmatStats(outl_patern,double(ind_out));
        ACC_EOD(n_ann,ind_T)=stats.precision(2)
        F1_EOD(n_ann,ind_T)=stats.Fscore(2)
        figure(151);clf;hold on;box on;
        ii=find(outl_patern==0);plot(A(1,ii),A(2,ii),'b.','MarkerSize',2);
        ii=find(outl_patern==1);plot(A(1,ii),A(2,ii),'bx','LineWidth',3,'MarkerSize',9);
        ii=find(ind_out==1);plot(A(1,ii),A(2,ii),'ro','LineWidth',2,'MarkerSize',6);
        xlabel('dimension 1','FontSize',FS); ylabel('dimension 2','FontSize',FS); 
        legend({'normally-distributed samples (\mu_d=0)','uniformly distributed outliers (\mu_d=3.5)','outliers detected with EOS'},'FontSize',18);
        title({['Outlier detection with Entropic Outlier Sparsification algorithm (EOS)']
        ['(T=' num2str(size(A,2)) ', D=' num2str(d) ', p=' num2str(prop) ')']});set(gcf,'Position',[10 100 1300  700]);
        set(gca,'FontSize',24,'LineWidth',2);


        err_mu_hadi(n_ann,ind_T)=norm(mu(1,:)-mean(A(:,1:round(T-prop*T))'),2)/d
        err_mu_EDO(n_ann,ind_T)=norm(mu_out(1,:)-mean(A(:,1:round(T-prop*T))'),2)/d
        err_V_hadi(n_ann,ind_T)=norm(S-cov(A(:,1:round(T-prop*T))'),2)/d
        err_V_EDO(n_ann,ind_T)=norm(V_out(:,:,1)-cov(A(:,1:round(T-prop*T))'),2)/d
        
        err_mu_hadi_exact(n_ann,ind_T)=norm(mu(1,:)-mean(A(:,1:round(T-prop*T))'),2)/d
        err_mu_EDO_exact(n_ann,ind_T)=norm(mu_out(1,:)-mean(A(:,1:round(T-prop*T))'),2)/d
        
        tic;
        [sig_rob,mu_rob,mah,outliers] = robustcov(A','Method','olivehawkins');
        t_rob(n_ann,ind_T)=toc;
        figure(152);clf;hold on;box on;
        ii=find(outl_patern==0);plot(A(1,ii),A(2,ii),'b.','MarkerSize',2);
        ii=find(outl_patern==1);plot(A(1,ii),A(2,ii),'bx','LineWidth',3,'MarkerSize',9);
        ii=find(outliers==1);plot(A(1,ii),A(2,ii),'ro','LineWidth',2,'MarkerSize',6);
        xlabel('dimension 1','FontSize',FS); ylabel('dimension 2','FontSize',FS); 
        legend({'normally-distributed samples (\mu_d=0)','uniformly distributed outliers (\mu_d=3.5)','outliers detected with OH'},'FontSize',13);
        title({['Outlier detection with Olive-Hawkins (OH) algorithm']
        ['(T=' num2str(size(A,2)) ', D=' num2str(d) ', p=' num2str(prop) ')']});set(gcf,'Position',[10 100 1300  700]);
        set(gca,'FontSize',24,'LineWidth',2);
        
        stats = confusionmatStats(outl_patern,double(outliers));
        ACC_rob(n_ann,ind_T)=stats.precision(2)
        F1_rob(n_ann,ind_T)=stats.Fscore(2)
        err_mu_rob(n_ann,ind_T)=norm(mu_rob-mean(A(:,1:round(T-prop*T))'),2)/d
        err_V_rob(n_ann,ind_T)=norm(sig_rob-cov(A(:,1:round(T-prop*T))'),2)/d

        tic;
        [sig_rob,mu_rob,mah,outliers] = robustcov(A','Method','ogk');
        t_rob_2(n_ann,ind_T)=toc;
        stats = confusionmatStats(outl_patern,double(outliers));
        ACC_rob2(n_ann,ind_T)=stats.precision(2)
        F1_rob2(n_ann,ind_T)=stats.Fscore(2)
        err_mu_rob_2(n_ann,ind_T)=norm(mu_rob-mean(A(:,1:round(T-prop*T))'),2)/d
        err_V_rob_2(n_ann,ind_T)=norm(sig_rob-cov(A(:,1:round(T-prop*T))'),2)/d
        figure(153);clf;hold on;box on;
        ii=find(outl_patern==0);plot(A(1,ii),A(2,ii),'b.','MarkerSize',2);
        ii=find(outl_patern==1);plot(A(1,ii),A(2,ii),'bx','LineWidth',3,'MarkerSize',9);
        ii=find(outliers==1);plot(A(1,ii),A(2,ii),'ro','LineWidth',2,'MarkerSize',6);
        xlabel('dimension 1','FontSize',FS); ylabel('dimension 2','FontSize',FS); 
        legend({'normally-distributed samples (\mu_d=0)','uniformly distributed outliers (\mu_d=3.5)','outliers detected with OGK'},'FontSize',13);
        title({['Outlier detection with Orthogonalized Gnanadesikan-Kettenring algorithm (OGK)']
        ['(T=' num2str(size(A,2)) ', D=' num2str(d) ', p=' num2str(prop) ')']});set(gcf,'Position',[10 100 1300  700]);
        set(gca,'FontSize',24,'LineWidth',2);
        
         tic;
         [forest,tf_forest,s_forest] = iforest(A(:,1:200)');
         [tf_forest,sTest_forest] = isanomaly(forest,A');
         t_for(n_ann,ind_T)=toc;
         stats = confusionmatStats(outl_patern,double(tf_forest));
         ACC_for(n_ann,ind_T)=stats.precision(2)
         F1_for(n_ann,ind_T)=stats.Fscore(2)
         idx=find(tf_forest==0);
         err_mu_for(n_ann,ind_T)=norm(mean(A(:,idx)')-mean(A(:,1:round(T-prop*T))'),2)/d
         err_V_for(n_ann,ind_T)=norm(cov(A(:,idx)')-cov(A(:,1:round(T-prop*T))'),2)/d
         figure(154);clf;hold on;box on;
         ii=find(outl_patern==0);plot(A(1,ii),A(2,ii),'b.','MarkerSize',2);
         ii=find(outl_patern==1);plot(A(1,ii),A(2,ii),'bx','LineWidth',3,'MarkerSize',9);
         ii=find(tf_forest==1);plot(A(1,ii),A(2,ii),'ro','LineWidth',2,'MarkerSize',6);
         xlabel('dimension 1','FontSize',FS); ylabel('dimension 2','FontSize',FS); 
         legend({'normally-distributed samples (\mu_d=0)','uniformly distributed outliers (\mu_d=3.5)','outliers detected with IRF'},'FontSize',13);
         title({['Outlier detection with Isolating Random Forests (IRF)']
         ['(T=' num2str(size(A,2)) ', D=' num2str(d) ', p=' num2str(prop) ')']});set(gcf,'Position',[10 100 1300  700]);
         set(gca,'FontSize',24,'LineWidth',2);

                  tic;
Mdl = fitcsvm(A(:,1:200)',ones(size(A(:,1:200)',1),1),OutlierFraction=0, ...
    KernelScale="auto",Standardize=true);
[~,sTest_OCSVM] = predict(Mdl,A');
tfTest_OCSVM = sTest_OCSVM < 0;
         t_svm(n_ann,ind_T)=toc;
         stats = confusionmatStats(outl_patern,double(tfTest_OCSVM));
         ACC_svm(n_ann,ind_T)=stats.precision(2)
         F1_svm(n_ann,ind_T)=stats.Fscore(2)
         idx=find(tfTest_OCSVM==0);
         err_mu_svm(n_ann,ind_T)=norm(mean(A(:,idx)')-mean(A(:,1:round(T-prop*T))'),2)/d
         err_V_svm(n_ann,ind_T)=norm(cov(A(:,idx)')-cov(A(:,1:round(T-prop*T))'),2)/d
         figure(155);clf;hold on;box on;
         ii=find(outl_patern==0);plot(A(1,ii),A(2,ii),'b.','MarkerSize',2);
         ii=find(outl_patern==1);plot(A(1,ii),A(2,ii),'bx','LineWidth',3,'MarkerSize',9);
         ii=find(tfTest_OCSVM==1);plot(A(1,ii),A(2,ii),'ro','LineWidth',2,'MarkerSize',6);
         xlabel('dimension 1','FontSize',FS); ylabel('dimension 2','FontSize',FS); 
         legend({'normally-distributed samples (\mu_d=0)','uniformly distributed outliers (\mu_d=3.5)','outliers detected with OCSVM'},'FontSize',13);
         title({['Outlier detection with One-Class SVM (OCSVM)']
         ['(T=' num2str(size(A,2)) ', D=' num2str(d) ', p=' num2str(prop) ')']});set(gcf,'Position',[10 100 1300  700]);
         set(gca,'FontSize',24,'LineWidth',2);

        %    id_out=double(RD_valid>chi_crt(4));
        %    stats = confusionmatStats(outl_patern,id_out);
        %    ACC(ind_T)=stats.precision(2);
        %    PREC(ind_T)=stats.precision(2);
        %    FSCORE(ind_T)=stats.Fscore(2);
    end
end

figure;EB(PROP,err_mu_rob,'b-');hold on;EB(PROP,err_mu_rob_2,'b:x');EB(PROP,err_mu_hadi,'gx:');EB(PROP,err_mu_for,'mx-');EB(PROP,err_mu_svm,'k<--');EB(PROP,err_mu_EDO,'ro--');set(gca,'YScale','log');
legend({'Olive-Hawkins Algorithm','Gnanadesikan-Kettenring Algorithm', 'Hadi Algorithm','Isolating Forest Algorithm', 'One-Class SVM', 'Entropic Outlier Detection (EOD)'},'FontSize',18,'Location','southwest')
         title({['Comparison of Error for Mean (T=' num2str(size(A,2)) ', D=' num2str(d) ')']});set(gcf,'Position',[10 100 1300  700]);
         xlabel('Non-Gaussian Anomaly Proportion, p');ylabel('l2-Error of Estimating the Mean'); 
         set(gca,'FontSize',24,'LineWidth',2,'XScale','linear');box on;

         figure;EB(PROP,ACC_rob,'b-');hold on;EB(PROP,ACC_rob2,'b:x');EB(PROP,ACC_hadi,'gx:');EB(PROP,ACC_for,'mx-');EB(PROP,ACC_svm,'k<--');EB(PROP,ACC_EOD,'ro--');set(gca,'XScale','log');
legend({'Olive-Hawkins Algorithm','Gnanadesikan-Kettenring Algorithm', 'Hadi Algorithm','Isolating Forest Algorithm', 'One-Class SVM', 'Entropic Outlier Detection (EOD)'},'FontSize',18,'Location','southwest')
         title({['Precision Comparison (T=' num2str(size(A,2)) ', D=' num2str(d) ')']});set(gcf,'Position',[10 100 1300  700]);
         xlabel('Non-Gaussian Anomaly Proportion, p');ylabel('Precision '); 
         set(gca,'FontSize',24,'LineWidth',2,'XScale','linear');box on;ylim([0 1.05]);

figure;EB(PROP,F1_rob,'b-');hold on;EB(PROP,F1_rob2,'b:x');EB(PROP,ACC_hadi,'gx:');EB(PROP,F1_for,'mx-');EB(PROP,F1_svm,'k<--');EB(PROP,F1_EOD,'ro--');set(gca,'XScale','log');
legend({'Olive-Hawkins Algorithm','Gnanadesikan-Kettenring Algorithm', 'Hadi Algorithm','Isolating Forest Algorithm', 'One-Class SVM', 'Entropic Outlier Detection (EOD)'},'FontSize',18,'Location','southwest')
         title({['F1-score Comparison (T=' num2str(size(A,2)) ', D=' num2str(d) ')']});set(gcf,'Position',[10 100 1300  700]);
         xlabel('Non-Gaussian Anomaly Proportion, p');ylabel('F1-score '); 
         set(gca,'FontSize',24,'LineWidth',2,'XScale','linear');box on;ylim([0 1.05]);

figure;EB(PROP,t_rob,'b-');hold on;EB(PROP,t_rob_2,'b:x');EB(PROP,t_hadi,'gx:');EB(PROP,t_for,'mx-');EB(PROP,t_svm,'k<--');EB(PROP,t,'ro--');set(gca,'XScale','log','YScale','log');
legend({'Olive-Hawkins Algorithm','Gnanadesikan-Kettenring Algorithm', 'Hadi Algorithm','Isolating Forest Algorithm', 'One-Class SVM', 'Entropic Outlier Detection (EOD)'},'FontSize',18,'Location','northwest')
         title({['Cost Comparison (T=' num2str(size(A,2)) ', D=' num2str(d) ')']});set(gcf,'Position',[10 100 1300  700]);
         xlabel('Non-Gaussian Anomaly Proportion, p');ylabel('CPU time, sec. '); 
         set(gca,'FontSize',24,'LineWidth',2,'XScale','linear');box on;


figure;loglog(DD,(err_mu_rob),'b-');hold on;loglog(DD,(err_mu_rob_2),'b:x');loglog(DD,(err_mu_hadi),'gx:');loglog(DD,(err_mu_for),'mx-');loglog(DD,(err_mu_svm),'k<--');loglog(DD,(err_mu_EDO),'ro--');
figure;loglog(DD,(err_V_rob),'b-');hold on;loglog(DD,(err_V_rob_2),'b:x');loglog(DD,(err_V_hadi),'gx:');loglog(DD,(err_V_for),'mx-');loglog(DD,(err_V_svm),'k<--');loglog(DD,(err_V_EDO),'ro--')
%figure;loglog(DD,(t_rob),'b-');hold on;loglog(DD,(t_rob_2),'b:x');loglog(DD,(t_hadi),'gx:','LineWidth',2);loglog(DD,(t_for),'mx-');loglog(DD,(t_svm),'k<--');loglog(DD,(t),'ro--')
%figure;semilogx(DD,(ACC_rob),'b-');hold on;semilogx(DD,(ACC_rob2),'b:x');semilogx(DD,(ACC_hadi),'gx:','LineWidth',2);semilogx(DD,(ACC_for),'mx-');semilogx(DD,(ACC_svm),'k<--');semilogx(DD,(ACC_EOD),'ro--')
%figure;semilogx(DD,(F1_rob),'b-');hold on;semilogx(DD,(F1_rob2),'b:x');semilogx(DD,(F1_hadi),'gx:','LineWidth',2);semilogx(DD,(F1_for),'mx-');semilogx(DD,(F1_svm),'k<--');semilogx(DD,(F1_EOD),'ro--')

%figure;loglog(DD,mean(err_mu_rob),'b-');hold on;loglog(DD,mean(err_mu_rob_2),'b:x');loglog(DD,mean(err_mu_hadi),'gx:');loglog(DD,mean(err_mu_EDO),'ro--')
%figure;loglog(DD,mean(err_V_rob),'b-');hold on;loglog(DD,mean(err_V_rob_2),'b:x');loglog(DD,mean(err_V_hadi),'gx:');loglog(DD,mean(err_V_EDO),'ro--')
%figure;loglog(DD,mean(t_rob),'b-');hold on;loglog(DD,mean(t_rob_2),'b:x');loglog(DD,mean(t_hadi),'gx:');loglog(DD,mean(t),'ro--')

%figure;loglog(DD,t,'ko--')
%figure;plot(A_valid(1,:),A_valid(2,:),'.');hold on;plot(A_valid(1,id_out),A_valid(2,id_out),'ro');
