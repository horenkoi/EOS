function [] = EB(DD,X,sym)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
X(X==0) = NaN;
D=size(X,1);
for i=1:size(X,2)  
    if sum(isnan(X(:,i)))<D
      pd = fitdist(X(:,i),'Normal');
      ci = paramci(pd);
      mu(i)=pd.mu;
      pl(i)=ci(1,1)-mu(i);
      mi(i)=mu(i)-ci(2,1);
    else
      mu(i)=NaN;
      pl(i)=0;
      mi(i)=0;
    end
end
hold on;errorbar(DD,mu,mi,pl,sym,'LineWidth',2);
end