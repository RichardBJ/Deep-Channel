function [minx, maxx,YLim]=fullplot(app)
    minx=min(app.data(:,1));
    maxx=max(app.data(:,1));          
    plot(app.rawAxis,app.data(:,1),app.data(:,2));
    YLim=[0,max(app.data(:,3)*1.25)];    
        if ((app.TwoPlot.Value==true) || (app.ThreePlot.Value==true))
            plot(app.idealAxis,app.data(:,1),app.data(:,3));            
        end
        if app.ThreePlot.Value==false
            [raw,idl]=calcCombined(app);
            plot(app.extraAxis,raw(:,1),raw(:,2));
            hold(app.extraAxis,'on');
            plot(app.extraAxis,idl(:,1),idl(:,2));    
            hold(app.extraAxis,'off');
        end

end