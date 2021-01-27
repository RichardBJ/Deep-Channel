close all
clear all
sec=10;% Number of seconds to produce
dt=1/10000; %seconds
%samples needs to be enough to be generating long enough strips
%but it's length is stoichastic!
samples=10000;
samplesout=int64(sec/dt);
maxchannels=5;
maxstates=6;
maxrecords=1000; % 10000 here?
%time=linspace(0,samples-1,samples); BUG!!?
time=0:samples-1;
filename='states3n.xlsx';
scheme= xlsread(filename,1,'B2:G7');
isopen= xlsread(filename,1,'B8:G8');
state=1;
for record=1:maxrecords    
   disp(["Record =",num2str(record)]);
%% Multichannels
    out=zeros(samplesout,2);
    channels=0;
    shorty=0;

    while channels<maxchannels
        channels=channels+1;
        [temp,state]=fstatesn(scheme,isopen,dt,samples,time,state,maxstates);
        if length(temp)>samplesout
            shorty=0; %reset the time saver variable shorty.
            if channels ==1
                out(1:samplesout,:)=temp(1:samplesout,:);
            else
                out(1:samplesout,2)=out(1:samplesout,2)+temp(1:samplesout,2);
            end
            
           samples=int64(0.9*samples); %may be wasting time to go with shorter search
           say=["samples too long at = ",num2str(samples)];
           disp(say);
           drawnow('update');

        else
            shorty=shorty+1;
            if shorty>5
                shorty=0;
                samples=int64(1.5*samples); %we didn't get enough samples so increase and go again.
                %now do this channel number again ignore warning!
            end   
            say=["samples too short at = ",num2str(samples)];
            disp(say);
            drawnow('update');
            channels=channels-1; 
        end
    end
    %% Crops of the first 2 seconds because of the bug I cannot trace...
 
  
    %dlmwrite(['astr' num2str(record) '.csv'],out,'precision','%10.10f');
    csvwrite(['astr' num2str(record) '.csv'],out(:,2));

%     drawnow 

end
plot(out(:,1),out(:,2));
ylim([0,maxchannels+1])

function [out,state]=fstatesn(scheme,isopen,dt,samples,time,state,maxstates)  
%C-O-O
    %READ FROM A TABLE!!
    % State 1=colA
    % state 2
    % startstate=1+int64(rand(1,1)*(maxstates-1));
    [disters,commonest]=getalldists(maxstates,scheme,samples); 
    
    %% THESE NEXT LINES MUST CHANGE IF MAXSTATES INCREASE FROM 5
    open=zeros(1,maxstates);
    for event=1:maxstates
       if isopen(event)==1
           open(event)=event;
       end
    end
    open=open(open>0);
    lifetimes=[];
    %states=[];
    for ii=1:samples
        %% look up the correct range of kf and kb on the basis of the current state
        tl=getlifetime(ii,disters,state);
        state=nextstate(state, maxstates, scheme);
        %states=[states;state];
        %disp(state);
        if ismember(state,open)
            lifetimes=[lifetimes;[tl,1]]; 
        else
            lifetimes=[lifetimes;[tl,0]]; 
        end
    end
out=maketimeseries(lifetimes,dt);
state=commonest;
% out=[time',out'];
end

function state=nextstate(state,maxstates,scheme)
%     This was my original idea, but a similar one is here: 
%     http://www.scholarpedia.org/article/Stochastic_models_of_ion_channel_gating
    trans=neibours(state,maxstates,scheme);
    
    %% Now MLG from the above site for massive speed improvement.    
   if length(trans)==1
       state=trans(1,1);
   else
       psum=sum(trans(:,2));
       ran=rand(1)*psum;
       for row=1:length(trans(:,1))
           smmer=sum(trans(1:row,2));
           if (smmer>=ran)
               state=trans(row,1);
               return
           end
       end
   end   
end

function trans=neibours(lstate,maxstates,scheme)
    trans=[];
    for j=-maxstates:maxstates
       if (0<(lstate+j))&& ((lstate+j)<=maxstates)
           if scheme(lstate,lstate+j)>0
             %% sort out the preallocation for speed
             row=lstate;
             col=lstate+j;
             val=scheme(row,col);
             trans=[trans;double(col),val];
             if trans(1,2)==0
                debugpoint=1; disp ("bug down here should never be no possible transitisions");
             end
           end
       end
    end
%disp(trans); 
end

function [alldist,commonest]=getalldists(maxstates,scheme,samples)
    alldist={maxstates+1};
    
    for lstate=1:maxstates
    trans=[];
        for j=-maxstates:maxstates
           if (0<(lstate+j))&& ((lstate+j)<=maxstates)
               if scheme(lstate,lstate+j)>0
                 %% sort out the preallocation for speed
                 row=lstate;
                 col=lstate+j;
                 val=scheme(row,col);
                 trans=[trans;lstate,val];
               end
           end
        end       
    f=exprnd((1/sum(trans(:,2))),[samples,1]);
    %f(randperm(length(f))); unecessary
    alldist{lstate}=f;
    end
highest=0;
for ii=1:length(alldist)
    if mean(alldist{ii})>highest
        commonest=ii;
        highest=mean(alldist{ii});
    end
end
% str1=["commonest=",num2str(commonest)];
% disp(str1);
end

function thislife=getlifetime(nth,dists,state)
% disp(nth);
thisdis=dists{state};
thislife=thisdis(nth);  
end

function output=maketimeseries(lifetimes,dt)
    output=zeros(10000,2); %Needs to be more than samples in length
    cur=1;
    tlen=0;
    for jj=1:length(lifetimes)
        thislen=lifetimes(jj,1);
        thisstate=lifetimes(jj,2);
        num=round(thislen/dt);
        tlen=num+tlen;
        t1=max(output(:,1))+dt;
        t2=t1+thislen;
        a=linspace(t1,t2,num)';
        output(cur:cur+num-1,1)=a;
        output(cur:cur+num-1,2)=thisstate;
        cur=cur+num;
    end
%     output=output(1:tlen,:);
end