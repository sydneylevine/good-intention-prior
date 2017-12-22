function [W,Of,Oa,NP,AP,LF] = BinoBF2(N1,K1,N2,K2,Inc,SupFig)
% Computes the Bayes Factor for the comparison of two proportions.
% Syntax [W,Of,Oa,NP,AP,LF] = BinoBF2(N1,K1,N2,K2,Inc)
% N1, K1, N2 & K2 are the two numbers of trials (N1 and N2) and two numbers of
% "successes" (K1 and K2). In these instructions, K1/N1 is referred to as
% the observed proportion of the control group and K2/N2 is referred to 
% as the observed (or expected) proportion of the experimental group.

%       Inc specifies the alternative prior.  You can
%       choose to specify Inc in one of two ways:
%
%     1) Set Inc to be a 2D vector.  In this case the values of the vector are taken
%        to be the A & B parameters of the betapdf function. 

%         What is the betapdf function and why would you want to use it as
%         your alternate prior?  The beta distribution is a very handy
%         distribution when working with proportions.  First of all, the
%         support of the distribution falls between 0 and 1.  Second, the
%         distribution has two parameters: number of successes (A) and
%         number of failures (B).  So you can see why specifying an 
%         alternate prior in terms of the A and B parameters of a beta
%         distribution is very natural.  It is also instructive to note
%         that the betapdf function is the conjugate prior for the binomial 
%         likelihood function, which means that the posterior distribution,
%         which is the product of the prior and the likelihood function, is
%         itself a beta distribution. This property--beta prior->beta
%         posterior--is unique to the beta distribution.

%         To use this option, write the Inc term as [X Y] where X is the A 
%         parameter of the beta distribution and Y is the B parameter. 
%
%     2) Set Inc to be a signed scalar between -1 and 1.  

%       Should you use a positive or a negative value for Inc?  If you want your  
%       alternate prior to specify an "effect size" greater than the 
%       observed proportion of the control group, p, then the sign of Inc 
%       should be positive.  If you want your alternate prior to specify an
%       "effect size" less than p, then the sign of Inc should be negative.
%       If the sign of Inc is positive, then the lower limit on the effect
%       size is 0 (no decrease in p).  If the sign of Inc is negataive, 
%       then the upper limit of the effect size is 0 (no increase in p).
%     
        %       IMPORTANT NOTE on the phrase "effect size":  We are using "effect size" 
        %       to refer to the difference between the observed control group proportion, p, and the
        %       expected experimental proportion (K2/N2).  Put another way, "effect
        %       size" just refers to how big of a change in the proportion we
        %       expect given the experimental treatment.  Traditionally, the term
        %       "effect size" refers to z-scores, which is the expected or
        %       observed difference in the means normalized by the standard
        %       deviation of the control group (or by the pooled standard
        %       deviation)

%     What value of Inc should you specify?  Larger absolute values of Inc
%     indicate a narrower possible effect.  Specify a larger value of Inc
%     if you have reason to think that the effect should only fall in a
%     very particular range that is fairly close to p (the observed 
%     proportion of the control group).  Smaller absolute values of Inc 
%     indicate a wider possible range.  

%     Examples: Say you want to create an alternate prior based on the 
%     hypothesis that the effect on the experimental group could fall 
%     anywhere above the effect you observed in the control group (K1/N1).  
%     That is, you expect some increase in p, but have no idea about how
%     big or small that increase will be.  In that case, you want to specify a 
%     value of Inc that is positive and very small.  Inc of 0.01 assumes that the
%     effect could range from no difference (the expected value of K2/N2 is equal to K1/N1)
%     to maximum possible difference (expected value of K2/N2 = 1).
%       
%     What if you want to create an alterate prior that suggests that 
%     the effect could fall anywhere BELOW the effect you actually observed?  
%     Inc of -0.01 assumes that the effect could range from no difference
%     to maximum possible difference(expected value of K2/N2 = 0).

%     More specifics about Inc:
%     If Inc is negative: Inc*(K1/N1) = lower limit of effect size; upper
%     limit is 0
%     If Inc is positive: 1 - Inc*Q = upper limit of effect size, where Q =
%     1 - K1/N1; lower limit is 0

%     More examples:
%     If K1/N1 = .5 and Inc = -.5, then the upper limit on the size of the
%     effect is 0 (no increase in p) and the lower limit is .25. If Inc =
%     .5, then the lower limit on the size of the effect is 0 (no decrease
%     in p) and the upper limit is .75. If K1/N1 = .3 and Inc = -.8, then
%     the lower limit on the size of the effect is .8*.3 = .24. If Inc =
%     .7, then the upper limit on the size of the effect is 1 -.7(1-.3) =
%     .51.
%
%      If Inc is omitted, the alternate prior is uniform on the interval
%      [0 1]. This is equivalent to assuming a betapdf prior with A=B=1
%
% W is the weight of the evidence (the common log of the
% Bayes Factor); positive values indicate that the odds favor the null;
% negative indicate odds in favor of the alternative.
%
% Of is the odds in favor of the null;
%
% Oa is the odds against the null;
%
% NP is the null prior (the rescaled likelihood function for k1/N1);
% AP is the alternate prior.
%
% If no Inc is specified, then the alternative prior is uniform on the
% interval [0 1]
%
%SupFig suppresses the figure if it is true
%%
p = linspace(0,1,200);

dp = p(2) - p(1); % delta p (spacing increment)

NP =betapdf(p,K1+1,N1-K1+1); % posterior distribution - null prior
%%
if nargin<6;SupFig = false;end

if nargin<5 % if increment prior not specified
    
    AP = betapdf(p,1,1); % uniform on the interval [0 1]
    
elseif length(Inc)>1 % if Inc specifies an alternate prior by giving the
    % parameters of a betapdf
    
    AP = betapdf(p,Inc(1),Inc(2));
    
else
    
    [M,R] = max(NP); % mode of null prior = maximum likelihood estimate
    % of p value in first sample
    
    if Inc < 0
        
        LL = abs(Inc)*p(R); % Lower limit on IP
        
        NPt = NP/sum(NP); % normalizing in prep for convolution
        
        pa = linspace(-1,0,200); % backwards p values for IP (increment prior)
        
        LL1 = pa(find(pa>=LL-p(R),1));
        
        IP = unifpdf(pa,LL1,0);
        
        IP = IP/sum(IP); % normalizing
        
        A = [[pa(1:end-1) p]' [zeros(1,199) NPt]' [IP zeros(1,199)]'];
        
        F = ConvDists_local(A);
        
        AP = F((F(:,1)>=0)&(F(:,1)<=1),2);       
        
        AP = AP/(dp*trapz(AP)); % renormalizing
        
    else
        Q = 1 - p(R); % complementary probability
        
        UL = 1-Inc*Q; % Upper limit on IP
        
        NPt = NP/sum(NP); % normalizing in prep for convolution
        
        
        IP = unifpdf(p,0,UL - p(R));
        
        IP = IP/sum(IP); % normalizing
        
        A = [p' NPt' IP'];
        
        F = ConvDists_local(A);
        
        AP = F(F(:,1)<=1,2); 

        
        AP = AP/(dp*trapz(AP)); % renormalizing
        
    end
    
end

LF = betapdf(p,K2+1,N2-K2+1); % this is actually the posterior distribution
% function assuming an uninformative (uniform) prior, but that is just
% a rescaled version of the likelihood function, and the scale of the
% liklelihood function does not matter.

Of = trapz(NP.*LF)/trapz(reshape(AP,size(NP)).*LF);

Oa = 1/Of;

W = log10(Of);

if SupFig;return;end

figure

[Ax,Hyy(1),Hyy(2)] = plotyy(p,NP,p,LF);

xlabel('Proportion (p)','FontSize',18)

set(Hyy(1),'LineWidth',2)

set(Hyy(2),'LineWidth',2)

set(Ax(2),'LineWidth',2,'FontSize',18)

set(Ax(1),'LineWidth',2,'FontSize',18)

set(get(Ax(1),'YLabel'),'FontSize',18)

% set(get(Ax(2),'YLabel'),'FontSize',18)
% 
% set(get(Ax(1),'XLabel'),'FontSize',18)

hold on

Hyy(3) = plot(Ax(1),p,AP,'b--','LineWidth',2);

legend(Hyy,'Null Prior','Likelihood','Alt Prior','Location','Northwest')

ylabel(Ax(1),'Probability Density','FontSize',18)

ylabel(Ax(2),'Likelihood','FontSize',18)

function F = ConvDists_local(A)
% Syntax F = ConvDists_local(A)
% The array A is a common x-axis (first column) and two vectorized
% distribution functions defined over that axis (columns 2 and 3).
% The function returns the convolution of the two distributions in F.
% Col 1 of F is the expanded x axis, whose length is 1 less than twice the
% length of the x-axis vector in A; Col 2 gives the probability densities
% of the convolution distribution, normalized so that they sum to 1. The
% 2nd & 3rd cols of A must each sum to 1

% The convolution for row vectors u & v as functions of x looks like this
%
%	v1	v2	v3	v4	v5	 = A(:,3)		
%	u1	u2	u3	u4	u5	 = A(:,2)			
%	x1	x2	x3	x4	x5   = A(:,1)				
%									
%	u1v1	u1v2	u1v3	u1v4	u1v5				
%           u2v1	u2v2	u2v3	u2v4	u2v5			
%                   u3v1	u3v2	u3v3	u3v4	u3v5		
%                           u4v1	u4v2	u4v3	u4v4	u4v5	
%                                   u5v1	u5v2	u5v3	u5v4	u5v5
% F(:,2)=sums of the above columns:
%	F(:,1)= x1+x1	x1+x2	x1+x3	x1+x4	x1+x5	x2+x5	x3+x5	x4+x5	x5+x5
% 
%
% Notice that the first row is the vector v, scaled by u1; the second row
% is the vector v scaled by u2 and shifted one step to the right over the x
% axis; the third row is the vector v scaled by u3 and shifted 2 steps; and
% so on. Each scaled copy of v1 begins at the locus of the scaling factor
% in u.
%
% For viewing convenience, the function works with column vectors rather than
% row vectors. 

if (sum(A(:,2))<.99999)||(sum(A(:,2))>1.0001)||(sum(A(:,3))<.99999)||...
        (sum(A(:,3))>1.0001)
    
    disp('One or both input distributions do not sum to 1')
    
    F=[];
    
    return
    
end

F(:,1) = [A(1,1)+A(:,1);A(end,1)+A(2:end,1)]; % the expanded x axis

F(:,2) = conv(A(:,2),A(:,3)); % convolving the y vectors

F(:,2) = F(:,2)/sum(F(:,2)); % normalizing the convolution (rescaling it 
% so that it sums to 1)



