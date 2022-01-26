function [w_out,N_iter,mu_out,V_out,L_train,ind]=EDO(x,EPS,N_ann,x_valid)

TOL=1e-6;
MaxIter=50;


[D,T]=size(x);
T_valid=size(x_valid,2);

for i_eps=1:length(EPS)
    eps=EPS(i_eps);
    for i_ann=1:N_ann
        w=rand(1,T);
        mu=mean(x')';
        V=inv(cov(x'));
        invV=inv(V);
        
        w=w./sum(w);
        iter=1;
        tol=100000;
        diff=zeros(D,T);
        maholanobis_distance=zeros(1,T);
        while iter<=MaxIter
            %
            mu=sum(w.*x,2);
            %[LL,LogL]=LogLik(w,mu,V,invV,D,T,eps);
            %LL
            %mu=sum(w.*x,2);
            
            V=zeros(D);
            for t=1:T
                diff(:,t)=(x(:,t)-mu);
                V=V+w(t)*(diff(:,t)*diff(:,t)');
            end
            invV=inv(V);
            %invV=pinv(V);invV=0.5*(invV+invV');
            [LL,LogL]=LogLik(w,mu,V,invV,D,T,eps);
            %LL
            %w_old=w;
            w=EvaluateW(LogL,eps);
            [LL,LogL]=LogLik(w,mu,V,invV,D,T,eps);
            figure(i_eps);plot(w,'b:o');pause(0.1)
            %LL
            if iter==1
                L=LL;
            else
                L=[L LL];
                diffL=L(iter-1)-L(iter);
                if diffL<TOL
                    break
                end
            end
            iter=iter+1;
        end
        
%        diff_valid=zeros(D,T_valid);
%        maholanobis_distance_valid=zeros(1,T_valid);
%        for t=1:T_valid
%            diff_valid(:,t)=(x_valid(:,t)-mu);
%            maholanobis_distance_valid(t)=diff_valid(:,t)'*invV*diff_valid(:,t);
%        end
%        LogL_valid=1/D*(D/2*log(2*pi)+0.5*log(det(V))+0.5*maholanobis_distance_valid);
%        w_valid=EvaluateW(LogL_valid,eps);
        ind_out=(w<(1/T)*0.01);
%        L_valid=sum(w_valid.*LogL_valid);
        if i_ann==1
            w_out(i_eps,:)=w;
            ind(i_eps,:)=ind_out;
            %L_out(i_eps)=L(length(L));
            L_train(i_eps)=L(length(L));
            N_iter(i_eps)=iter;
            mu_out(i_eps,:)=mu';
            V_out(:,:,i_eps)=V;
        elseif L(length(L))<L_train(i_eps)
            w_out(i_eps,:)=w;
            ind(i_eps,:)=ind_out;
            %L_out(i_eps)=L_valid;
            N_iter(i_eps)=iter;
            L_train(i_eps)=L(length(L));
            mu_out(i_eps,:)=mu';
            V_out(:,:,i_eps)=V;
        end
    end
end


    function [LL,LogL]=LogLik(w,mu,V,invV,D,T,eps)
        diff=zeros(D,T);
        maholanobis_distance=zeros(1,T);
         %invV=inv(V);
        
        for t=1:T
            diff(:,t)=(x(:,t)-mu);
            maholanobis_distance(t)=diff(:,t)'*invV*diff(:,t);
        end
        
        LogL=1/D*(D/2*log(2*pi)+0.5*log(det(V))+0.5*maholanobis_distance);
        LL=sum(w.*LogL)+eps*sum(w.*log(max(1e-20,w)));
        
    end
    function [W]=EvaluateW(b,eps)
        z=exp(-b./(eps));
        W=z./sum(z);
    end
end
