% MDP with directed coordination graph,  ZOO
% Truncated local value functions
% Distributed RL, oi=[m_{I_i},di], both one-point and two-point feedbacks


clear all;
% close all;

N=100; gamma=0.8; T=10; eta=.05; r=2; K=500;

Ni=4; % number of agents in each cluster

kappa=1; % The index kappa

A=0.5; phi=0;

lay=5; % The number of feature parameters

H=10; % H is the times of samples at each episode

%undirected coordination graph
G=zeros(N);
for i=1:N
    if mod(i,2)==1
        if i==1
            G(1,2)=1; G(1,N)=1;
        end
        if i~=1
            G(i,i+1)=1;G(i,i-1)=1;
        end
    end
end

G=G+G';

n=100/Ni; % number of clusters


for i=1:n
    LVF(i).V=Ni*(i-1)+1:1:Ni*i;
    y=zeros(1,2*kappa);
    for k=1:kappa
        y(k)=Ni*(i-1)-kappa+k;
        y(kappa+k)=Ni*i+k;
    end

    for j=1:2*kappa
        if y(j)<1
            y(j)=y(j)+N;
        end
        if y(j)>N
            y(j)=y(j)-N;
        end
    end
    
    LVF(i).index= [y(1:kappa) LVF(i).V y(kappa+1:2*kappa)];
end

for l=1:n
    GG=zeros(N);
    GG(LVF(l).index,LVF(l).index)=G(LVF(l).index,LVF(l).index);
    deg=zeros(N,1); % degree of agent i
for i=1:N
    deg(i)=sum(GG(i,:));
end
matrix(l).C=zeros(N);
    for i=1:N
        for j=1:N
            if i~=j&&GG(i,j)~=0
              matrix(l).C(i,j)=1/(1+max(deg(i),deg(j)));
            end
        end
        if ismember(i,LVF(l).index)
        matrix(l).C(i,i)=1-sum(matrix(l).C(i,:));
        end
    end
end



ind=zeros(N); % determine the position of theta_{ij} in theta vector 
dp=0; % dimension of the global policy

Ii=zeros(N,1); 
for i=1:N
    Ii(i)=sum(G(:,i))+1;
    Proj(i).M=zeros(Ii(i),N); % extract partial observation from global state
    k=1;
    while k<=Ii(i)
        for j=1:N
            if G(j,i)~=0||j==i
                Proj(i).M(k,j)=1;
                k=k+1;
            end
            if G(i,j)~=0||i==j
            dp=dp+1;
            ind(i,j)=dp;
            end
        end
    end
end


Gc=digraph(G);
Gl=zeros(N);
for i=1:N
    for j=1:N
        if isempty(shortestpath(Gc,i,j))==0
            Gl(j,i)=1;
        end
    end
end






% theta(:,1)=ones(N^2,1);
% A=ones(N,1)*0.5; phi=ones(N,1);




for i=1:N
    feature(i).c=normrnd(0,1,Ii(i)+1,lay); % feature parameters for agent i
    for j=1:lay
        feature(i).c(1:Ii(i),j)=feature(i).c(1:Ii(i),j)/norm(feature(i).c(1:Ii(i),j))*(Ii(i));
        feature(i).c(Ii(i)+1,j)=0.1/(lay-1)*(j-1);
    end
end

    
z=zeros(N);
a=zeros(N); 
d=zeros(N,1); %local demands

% m00=0.1*[10, 10, 4, 3, 10, 8, 5, 4, 6]'; 

m00=ones(N,1);


%% Compute the value function with initial policy theta


sigma=0.01; % variance of the noise w

for i=1:lay
    theta(i).vec=zeros(dp,1);
end
[J,~]=Evaluatetheta(theta,gamma,m00,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF);



Jhg=zeros(K,H); Jhg(1,:)=ones(1,H)*J; % Record updated value based on global value evaluation
Jhl=zeros(K,H); Jhl(1,:)=ones(1,H)*J; % Record updated value based on local value evaluation

Jhg2=zeros(K,H); Jhg2(1,:)=ones(1,H)*J; % Global two-point feedback
Jhl2=zeros(K,H); Jhl2(1,:)=ones(1,H)*J; % Local two-point feedback


Jg=zeros(K,1); Jl=zeros(K,1);
Jg(1)=J; Jl(1)=J;

Jg2=zeros(K,1); Jl2=zeros(K,1);
Jg2(1)=J; Jl2(1)=J;

%% Run the RL algorithm

sampu=zeros(K-1,1);

for h=1:H
    
    for i=1:lay
    thetag(i).vec=zeros(dp,1);
    thetal(i).vec=zeros(dp,1);
    
    thetag2(i).vec=zeros(dp,1);
    thetal2(i).vec=zeros(dp,1);
    end

for k=1:K-1
%     m0=0.1*[10, 10, 4, 3, 10, 8, 5, 4, 6]'; 
   m0=zeros(N,1);
    for i=1:N
        w0=normrnd(0,0.01);
        if w0>=0.01
            w0=sign(w0)*0.01;
        end
        m0(i)=m00(i)+w0; % s0. How to ensure m0 to be positive?
    end
    
v=zeros(dp,1);


[Jc,~]=Evaluatetheta(thetag2,gamma,m0,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF); % Evaluate the current global policy theta

[~,Vald]=Evaluatetheta(thetal2,gamma,m0,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,0,LVF); % Evaluate the current local policy theta



    
u=normrnd(0,0.5,dp,lay); % Global sample
v=v+u;

for i=1:lay
    thetagp(i).vec=thetag(i).vec+r*u(:,i); % Perturbed policy
    thetalp(i).vec=thetal(i).vec+r*u(:,i); % Perturbed policy
    
    thetag2p(i).vec=thetag2(i).vec+r*u(:,i); % Perturbed two-point global policy
    thetal2p(i).vec=thetal2(i).vec+r*u(:,i);  %
end



[Jgp,~]=Evaluatetheta(thetagp,gamma,m0,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF); % Evaluate the perturbed policy thetap, obtain global value J    
   
[~,Vallp]=Evaluatetheta(thetalp,gamma,m0,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,0,LVF); % Evaluate the perturbed policy thetap, obtain local value J_i 
 
[Jg2p,~]=Evaluatetheta(thetag2p,gamma,m0,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF); % Evaluate the perturbed policy thetap, obtain global value J    
   
[~,Vall2p]=Evaluatetheta(thetal2p,gamma,m0,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,0,LVF); % Evaluate the perturbed policy thetap, obtain local value J_i 
 
gg=Jgp*u/r;

 gg2=(Jg2p-Jc)*u/r; % two-point zero-order oracle

gl=zeros(dp,lay);
gl2=zeros(dp,lay);
kk=1;
for i=1:N
    for j=1:N
        if G(i,j)~=0||i==j
            gl(kk,:)=Vallp(i)*u(kk,:);% one-point zero-order oracle based on local value
            gl2(kk,:)=(Vall2p(i)-Vald(i))*u(kk,:);% two-point zero-order oracle based on local value
            kk=kk+1;
        end
    end
end


for i=1:lay
    thetag(i).vec=thetag(i).vec+eta*gg(:,i); % Update policy based on global evaluation
    thetal(i).vec=thetal(i).vec+eta*gl(:,i); % Update policy based on local evaluation
    
    thetag2(i).vec=thetag2(i).vec+eta*gg2(:,i); % Update policy based on global evaluation
    thetal2(i).vec=thetal2(i).vec+eta*gl2(:,i); % Update policy based on local evaluation
end

Jhg(k+1,h)=Evaluatetheta(thetag,gamma,m00,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF); % Evaluate the updated policy Thetag
Jhl(k+1,h)=Evaluatetheta(thetal,gamma,m00,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF); % Evaluate the updated policy Thetal 

Jhg2(k+1,h)=Evaluatetheta(thetag2,gamma,m00,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF); % Evaluate the updated policy Thetag
Jhl2(k+1,h)=Evaluatetheta(thetal2,gamma,m00,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,1,LVF); % Evaluate the updated policy Thetal 

end



end

for k=1:K-1
Jg(k+1)=sum(Jhg(k+1,:))/H; 
Jl(k+1)=sum(Jhl(k+1,:))/H; 
Jg2(k+1)=sum(Jhg2(k+1,:))/H; 
Jl2(k+1)=sum(Jhl2(k+1,:))/H; 
end



y1=zeros(1,K); y2=zeros(1,K);
for k=1:K
    y1(k)=max(Jhg(k,:));
    y2(k)=min(Jhg(k,:));
end

xx=1:K;

figure;
% hold on 

color11='[0 0.4470 0.7410]';
% color12='[0.7137    0.9098    0.9804]'; %qian
% color12='[0.4275    0.8314    0.9882]'; %zhong
color12=color11;
shadedplot(xx,y1,y2,color12);

color21='[0.3490    0.5216    0.1961]';
% color22='[0.8627    0.9882    0.8863]';%'[0.6431    0.8784    0.6588]';
% color22='[0.7569    0.9608    0.7569]';
color22=color21;

y1=zeros(1,K); y2=zeros(1,K);
for k=1:K
    y1(k)=max(Jhl(k,:));
    y2(k)=min(Jhl(k,:));
end

xx=1:K;

hold on;
shadedplot(xx,y1,y2,color22)

   



color31='[0.4980    0.3176    0.6196]';
% color32='[0.7804    0.6706    0.8784]';
color32=color31;

y1=zeros(1,K); y2=zeros(1,K);
for k=1:K
    y1(k)=max(Jhg2(k,:));
    y2(k)=min(Jhg2(k,:));
end

xx=1:K;

hold on;
shadedplot(xx,y1,y2,color32)




color41='r';
% color42='[0.9098    0.5843    0.5843]';
color42=color41;

y1=zeros(1,K); y2=zeros(1,K);
for k=1:K
    y1(k)=max(Jhl2(k,:));
    y2(k)=min(Jhl2(k,:));
end

xx=1:K;

hold on;
shadedplot(xx,y1,y2,color42)

hold on;

h=plot(Jg,'color',color11,'LineWidth',2);

hold on;

h2=plot(Jl,'color',color21,'LineWidth',2); 

hold on;

h3=plot(Jg2,'color',color31,'LineWidth',2);

hold on;

h4=plot(Jl2,'color',color41,'LineWidth',2);

xlabel('Learning Episode','fontsize',20)
ylabel('W(\theta^k,\xi^k)', 'fontsize',20)
set(gca,'fontsize',20)
%  axis([0 K -3 -1])   
% title(['\eta= ', num2str(eta), ', r= ', num2str(r)])
title(['\kappa= ', num2str(kappa)])

legend([h,h2,h3,h4],'Centralized, one-point','Distributed, one-point','Centralized, two-point','Distributed, two-point')

save('Largecase')
savefig('Figure 1.fig')

function [Jp,Value]=Evaluatetheta(theta,gamma,m0,N,A,sigma,phi,T,G,Gl,feature,Proj,lay,ind,matrix,glo,LVF)
m=m0;
a=zeros(N);
d=zeros(N,1);

% for i=1:lay
%     theta(i).M=reshape(theta(i).vec,N,N); % Transform vector theta to matrix Theta
% end


LocaVal=zeros(N,1);

 for t=1:T 
 
     z=zeros(N);
    for i=1:N        % Induce the action based on policy and local observation
             
%         w=normrnd(0,sigma); % Normal distribution with mu=0,sigma^2
%         if abs(w)>=0.1
%             w=sign(w)*0.1;
%         end
        
        d(i)=A*sin(t);  % Local demand;  % Local demand
%          d(i)=0.4*d(i);
   
        obs(i).s=[Proj(i).M*m;d(i)];  % Local observation, including s_{I_i} and d_i
        
        sumzij=0;
        for j=1:N
            if G(i,j)~=0||i==j
                for l=1:lay
                    z(i,j)=z(i,j)+norm(obs(i).s-feature(i).c(:,l))^2*theta(l).vec(ind(i,j)); % z_{ij}
                end
                sumzij=sumzij+exp(-z(i,j));  % \sum_{j\in N_i}exp(z_{ij})
            end
        end
        
        for j=1:N
            if G(i,j)~=0
                a(i,j)=exp(-z(i,j))/sumzij; % a_{ij}
            end
        end
    end
       
       mp=m;
    
             
        for i=1:N
            
            if m(i)>=0
                R(i)=0;
            else
                R(i)=-m(i)^2;
            end
            LocaVal(i)=LocaVal(i)+gamma^(t-1)*R(i);
                        
            for j=1:N
                if G(i,j)~=0
                   m(i)=m(i)-a(i,j)*m(i);
                elseif G(j,i)~=0
                   m(i)=m(i)+a(j,i)*mp(j);
                end
            end
            m(i)=m(i)+d(i); 
        end
 end
 
 
 
Jp=sum(LocaVal);  % Global Value

Value=zeros(N,1); % Local value based on the learning graph




if glo==0 % Local value functions are required to be estimated

    Tc=10; n=numel(LVF);
    
    for l=1:n
        Vcl(l).mu=LocaVal; % Initial state of consensus
        for k=1:Tc
        Vcl(l).mu(:,k+1)=matrix(l).C*Vcl(l).mu(:,k); % Consensus
        end
        Vcl(l).V=Vcl(l).mu(:,k+1)*numel(LVF(l).index); % The estimated LVF for cluster l
        for i=1:N
            if ismember(i,LVF(l).V)  % if i belongs to cluster l
                Value(i)=Vcl(l).V(i);
            end
        end
    end


% Value1=zeros(N,1);
% for i=1:N
%     for j=1:N
%         for l=1:n
%         if ismember(i,LVF(l).V)&&ismember(j,LVF(l).index)  % including i itself
%             Value1(i)=Value1(i)+LocaVal(j);
%         end
%         end
%     end
% end

end

end
