function isit=istime(app)
    if length(app.data(:,1))<100
        disp('data too short ERRROOORRRRR!!!!')
    end
    last=0;
    trues=0;
    for i=1:100
        if app.data(i,1)>last
            trues=trues+1;
            last=app.data(i,1);    
        end  
    end

    if (trues>50) && (trues<99)
        disp ("ambiguious if time ERROR")
        
    elseif trues>=99
        isit=(app.data(100,1)-app.data(50,1))/50;
    else
        isit=0;
    end

end