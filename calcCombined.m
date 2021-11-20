function [raw, idl]=calcCombined(app)
    raw=[app.data(:,1),rescale(app.data(:,2))];
    idl=[app.data(:,1),rescale(app.data(:,3))];
end